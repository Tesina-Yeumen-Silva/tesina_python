# app/use_cases/process_pending_reports.py
import asyncio
from io import BytesIO
import aiohttp
from PIL import Image
from app.repositories import REPORT_STATES 
from app.config.logger import logger

class ProcessPendingReportsUseCase:
    def __init__(self, report_repo, clip_service, text_service, decision_service, clustering_service):
        self.repo = report_repo
        self.clip_service = clip_service
        self.text_service = text_service
        self.decision_service = decision_service
        self.clustering_service = clustering_service
        self.categories_list = list(text_service.semantic_map.keys())
        self.http_session = None 

    async def _get_session(self) -> aiohttp.ClientSession:
        """Inicializa la sesión HTTP de forma perezosa (lazy initialization)."""
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'})
        return self.http_session

    async def _download_image(self, url: str) -> Image.Image:
        """Descarga una imagen de internet de forma 100% asíncrona."""
        session = await self._get_session()
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Error al descargar imagen. Status: {response.status}")
            img_bytes = await response.read()
            return await asyncio.to_thread(lambda: Image.open(BytesIO(img_bytes)).convert('RGB'))

    async def execute(self, report_id: int):
        # 1. Traer el reporte específico
        report = await self.repo.get_report_by_id(report_id)
        if not report:
            logger.warning(f"No se encontró el reporte #{report_id} en la base de datos.")
            return

        # Obtener IDs de estados comunes de forma asíncrona
        id_rechazado, id_validado, id_duplicado = await asyncio.gather(
            self.repo.get_state_id_by_name(REPORT_STATES["RECHAZADO"]),
            self.repo.get_state_id_by_name(REPORT_STATES["VALIDADO"]),
            self.repo.get_state_id_by_name(REPORT_STATES["DUPLICADO"])
        )

        image_url = report['imageUrl']
        descripcion_usuario = report['description'] or ""

        logger.info(f"\n--- Evaluando Reporte #{report_id} ---")
        
        try:
            # 2. Descargar Imagen de forma asíncrona
            image = await self._download_image(image_url)

            # 3. Clasificación de Imagen (CLIP) en un hilo secundario para no bloquear
            clip_result = await asyncio.to_thread(self.clip_service.classify_image, image)
            if not clip_result["valid"]:
                logger.warning(f"CLIP Bloqueó la imagen: {clip_result['detail']}")
                await self.repo.add_history_entry_and_notify(
                    report_id, id_rechazado, REPORT_STATES["RECHAZADO"], clip_result['detail']
                )
                return

            # 4. Clasificación Semántica de Texto (MPNet) en hilo secundario
            text_scores = await asyncio.to_thread(self.text_service.classify_text, descripcion_usuario)
            text_decision = self.decision_service.get_best_text_category(text_scores)

            # 5. Fusión de decisiones (Texto vs Imagen)
            categoria_ia = None
            observacion_final = "Validado automáticamente por el motor de IA."

            if text_decision["valid"]:
                categoria_ia = text_decision["category"]
                logger.info(f"TEXTO GANADOR ({text_decision['confidence']:.2f}): {categoria_ia}")
            else:
                categoria_ia = clip_result["suggested_category"]
                observacion_final = f"Descripción ambigua. Categoría corregida visualmente a: {categoria_ia}."
                logger.info(f"IMAGEN GANADORA: Texto inválido/vacío. CLIP sugiere -> {categoria_ia}")

            # 6. Corregir Categoría en la Base de Datos si difiere
            id_categoria_ia = await self.repo.get_category_id_by_name(categoria_ia)
            if report['categoryId'] != id_categoria_ia:
                logger.info(f"Corrigiendo categoría ID: {report['categoryId']} -> {id_categoria_ia}")
                await self.repo.update_report_category(report_id, id_categoria_ia)

            # 7. Análisis Espacial de Duplicados (BallTree)
            logger.info("Buscando contexto geográfico en la base de datos...")
            historical_reports = await self.repo.get_recent_reports_by_category(
                category_id=id_categoria_ia, exclude_report_id=report_id, days=15
            )

            # Verificación por vecindad geográfica
            es_duplicado_geografico = self.clustering_service.is_duplicate(report, historical_reports)
            es_duplicado_real = False
            
            if es_duplicado_geografico:
                logger.info("Cercanía detectada. Iniciando peritaje visual iterativo...")
                
                for reporte_conflicto in historical_reports:
                    try:
                        image_hist = await self._download_image(reporte_conflicto['imageUrl'])
                        
                        # Comparación visual en hilo secundario
                        similitud = await asyncio.to_thread(
                            self.clip_service.compare_images, image, image_hist
                        )
                        logger.info(f"Similitud con reporte #{reporte_conflicto['id']}: {similitud:.2f}")
                        
                        if similitud >= 0.75:
                            logger.warning(f"CONFIRMADO: Duplicado real con reporte #{reporte_conflicto['id']}.")
                            es_duplicado_real = True
                            observacion_final = f"Reporte duplicado. (Cercanía espacial + Similitud visual con #{reporte_conflicto['id']}: {similitud*100:.1f}%)."
                            break 
                            
                    except Exception as e:
                        logger.warning(f"Error al comparar con reporte #{reporte_conflicto['id']}: {e}")
                        continue 

            # 8. Guardar Evento en el historial y Notificar
            if es_duplicado_real:
                await self.repo.add_history_entry_and_notify(
                    report_id, id_duplicado, REPORT_STATES["DUPLICADO"], observacion_final
                )
                logger.warning("Guardado como DUPLICADO")
            else:
                await self.repo.add_history_entry_and_notify(
                    report_id, id_validado, REPORT_STATES["VALIDADO"], observacion_final
                )
                logger.info("Guardado como VALIDADO")

        except Exception as item_error:
            logger.error(f"Error al procesar el reporte #{report_id}: {item_error}", exc_info=True)