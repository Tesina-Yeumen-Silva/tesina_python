# app/use_cases/process_pending_reports.py
import urllib.request
from io import BytesIO
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

    async def execute(self):
        # 1. Traer reportes que la IA tiene pendientes procesar
        pending_reports = await self.repo.get_pending_reports(REPORT_STATES["PENDIENTE"])
        if not pending_reports:
            logger.info("No se encontraron nuevos reportes para analizar.")
            return

        logger.info(f"Iniciando análisis multimodal para {len(pending_reports)} reportes.")

        # Obtener IDs de estados comunes para ahorrar consultas en el bucle
        id_rechazado = await self.repo.get_state_id_by_name(REPORT_STATES["RECHAZADO"])
        id_validado = await self.repo.get_state_id_by_name(REPORT_STATES["VALIDADO"])
        id_duplicado = await self.repo.get_state_id_by_name(REPORT_STATES["DUPLICADO"])

        for report in pending_reports:
            report_id = report['id']
            image_url = report['imageUrl']
            descripcion_usuario = report['description'] or ""

            logger.info(f"\n--- Evaluando Reporte #{report_id} ---")
            
            try:
                # 2. Descargar Imagen en memoria
                req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    image = Image.open(BytesIO(response.read())).convert('RGB')

                # 3. Clasificación de Imagen (CLIP)
                clip_result = self.clip_service.classify_image(image)
                if not clip_result["valid"]:
                    logger.warning(f"CLIP Bloqueó la imagen: {clip_result['detail']}")
                    
                    await self.repo.add_history_entry_and_notify(
                        report_id, 
                        id_rechazado, 
                        REPORT_STATES["RECHAZADO"], 
                        clip_result['detail']
                    )
                    continue

                # 4. Clasificación Semántica de Texto (MPNet)
                text_scores = self.text_service.classify_text(descripcion_usuario, self.categories_list)
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

                # 6. Corregir Categoría en la Base de Datos si la IA difiere del usuario
                id_categoria_ia = await self.repo.get_category_id_by_name(categoria_ia)
                if report['categoryId'] != id_categoria_ia:
                    logger.info(f"Corrigiendo categoría ID: {report['categoryId']} -> {id_categoria_ia}")
                    await self.repo.update_report_category(report_id, id_categoria_ia)

                # 7. DBSCAN: Análisis Espacial de Duplicados
                logger.info("Buscando contexto geográfico en la base de datos...")
                
                # Traemos los reportes históricos activos de esa misma categoría corregida
                historical_reports = await self.repo.get_recent_reports_by_category(
                    category_id=id_categoria_ia,
                    exclude_report_id=report_id,
                    days=15
                )

                es_duplicado_geografico = self.clustering_service.is_duplicate(report, historical_reports)
                es_duplicado_real = False
                
                if es_duplicado_geografico:
                    logger.info("DBSCAN detectó cercanía. Iniciando peritaje visual con IA...")
                    
                    # Asumimos el reporte con el que colisiona (el primero de la lista para este análisis)
                    reporte_conflicto = historical_reports[0]
                    
                    try:
                        # Descargamos la imagen del reporte histórico con el que colisiona
                        req_hist = urllib.request.Request(reporte_conflicto['imageUrl'], headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req_hist) as response_hist:
                            image_hist = Image.open(BytesIO(response_hist.read())).convert('RGB')
                        
                        # CLIP compara las similitudes visuales de ambas fotos
                        similitud = self.clip_service.compare_images(image, image_hist)
                        logger.info(f"Similitud visual calculada: {similitud:.2f}")
                        
                        # Si superan el 75% de similitud, confirmamos el duplicado real
                        if similitud >= 0.75:
                            logger.warning("CONFIRMADO: Las fotos muestran el mismo problema.")
                            es_duplicado_real = True
                            observacion_final = f"Reporte agrupado como duplicado. (Cercanía espacial + Similitud visual: {similitud*100:.1f}%)."
                        else:
                            logger.info("DESCARTADO: Están cerca, pero las fotos son muy distintas.")
                            es_duplicado_real = False
                            
                    except Exception as e:
                        logger.warning(f"Error al comparar imágenes. Se asume NO duplicado por seguridad. Error: {e}")
                        es_duplicado_real = False
                
                # 8. Guardar Evento en el historial y Notificar
                if es_duplicado_real:
                    await self.repo.add_history_entry_and_notify(
                        report_id, 
                        id_duplicado, 
                        REPORT_STATES["DUPLICADO"], 
                        observacion_final
                    )
                    logger.warning(f"Guardado como DUPLICADO")
                else:
                    await self.repo.add_history_entry_and_notify(
                        report_id, 
                        id_validado, 
                        REPORT_STATES["VALIDADO"], 
                        observacion_final
                    )
                    logger.info(f"Guardado como VALIDADO")

            except Exception as item_error:
                logger.error(f"Error al procesar de forma individual el reporte #{report_id}: {item_error}", exc_info=True)