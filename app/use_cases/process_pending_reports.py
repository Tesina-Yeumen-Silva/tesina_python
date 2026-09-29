# app/use_cases/process_pending_reports.py
import asyncio
from io import BytesIO
import aiohttp
from PIL import Image
from app.repositories import REPORT_STATES 
from app.config.logger import logger
import json
import os
from datetime import datetime, timezone

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

    async def close(self):
        """Cierra la sesión HTTP de forma limpia al apagar el worker."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

    async def _download_image(self, url: str) -> Image.Image:
        """Descarga una imagen de internet de forma 100% asíncrona."""
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                raise Exception(f"Error al descargar imagen. Status: {response.status}")
            img_bytes = await response.read()
            return await asyncio.to_thread(lambda: Image.open(BytesIO(img_bytes)).convert('RGB'))

    def _log_telemetry(self, data: dict):
        """Guarda los resultados crudos de los modelos de IA localmente en un JSONL."""
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        telemetry_file = os.path.join(logs_dir, "ai_telemetry.jsonl")
        
        try:
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error guardando telemetria: {e}")

    async def execute(self, report_id: int):
        # 1. Traer el reporte específico
        report = await self.repo.get_report_by_id(report_id)
        if not report:
            logger.warning(f"No se encontró el reporte #{report_id} en la base de datos.")
            return

        # Obtener IDs de estados comunes de forma asíncrona
        id_rechazado, id_validado, id_duplicado, id_pendiente = await asyncio.gather(
            self.repo.get_state_id_by_name(REPORT_STATES.get("RECHAZADO", "RECHAZADO")),
            self.repo.get_state_id_by_name(REPORT_STATES.get("VALIDADO", "VALIDADO")),
            self.repo.get_state_id_by_name(REPORT_STATES.get("DUPLICADO", "DUPLICADO")),
            self.repo.get_state_id_by_name(REPORT_STATES.get("PENDIENTE", "PENDIENTE"))
        )

        image_url = report['imageUrl']
        descripcion_usuario = report['description'] or ""

        logger.info(f"\n--- Evaluando Reporte #{report_id} ---")
        
        try:
            # 2. Descargar Imagen de forma asíncrona
            image = await self._download_image(image_url)

            # 3. Clasificación de Imagen (CLIP) y Texto (BETO/MPNet) en paralelo
            clip_result, text_scores = await asyncio.gather(
                asyncio.to_thread(self.clip_service.classify_image, image),
                asyncio.to_thread(self.text_service.classify_text, descripcion_usuario)
            )
            text_decision = self.decision_service.get_best_text_category(text_scores)

            # 4. Control de Validación con Soft Gating
            if not clip_result.get("valid", False):
                if clip_result.get("rejection_reason") in ["not_real_photo", "not_outdoor_urban", "inappropriate_content", "explicitly_no_problem"] or not text_decision.get("valid", False):
                    logger.warning(f"CLIP Bloqueó la imagen: {clip_result.get('detail')}")
                    await self.repo.add_history_entry_and_notify(
                        report_id, id_rechazado, REPORT_STATES.get("RECHAZADO", "RECHAZADO"), clip_result.get('detail', 'Rechazado')
                    )
                    
                    # Registrar telemetría del rechazo
                    self._log_telemetry({
                        "report_id": report_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "final_state": "RECHAZADO",
                        "vision_model": clip_result,
                        "text_model": text_decision
                    })
                    return
                else:
                    logger.info("Soft Gating activado: Imagen ambigua rescatada por validación textual.")

            # 5. Fusión ponderada de decisiones (Texto + Imagen)
            fusion_result = self.decision_service.fuse_decisions(text_scores, clip_result)
            categoria_ia = fusion_result["category"]
            motivo_decision = fusion_result["motivo"]
            logger.info(f"DECISIÓN FUSIÓN MULTIMODAL ({categoria_ia}): {motivo_decision}")

            # 6. Corregir Categoría en la Base de Datos si difiere
            id_categoria_ia = await self.repo.get_category_id_by_name(categoria_ia)
            categoria_modificada = False
            nombre_categoria_original = None

            if report['categoryId'] != id_categoria_ia:
                nombre_categoria_original = await self.repo.get_category_name_by_id(report['categoryId'])
                logger.info(f"Corrigiendo categoría ID: {report['categoryId']} ({nombre_categoria_original}) -> {id_categoria_ia} ({categoria_ia})")
                await self.repo.update_report_category(report_id, id_categoria_ia)
                categoria_modificada = True
            else:
                nombre_categoria_original = categoria_ia

            # Formar observación de validación
            if categoria_modificada:
                observacion_final = f"Validado automáticamente por el motor de IA. Categoría corregida de '{nombre_categoria_original or 'Desconocida'}' a '{categoria_ia}' según {motivo_decision}."
            else:
                observacion_final = f"Validado automáticamente por el motor de IA. ({motivo_decision})"

            # 7. Análisis Espacial de Duplicados (BallTree)
            logger.info("Buscando contexto geográfico en la base de datos...")
            historical_reports = await self.repo.get_recent_reports_by_category(
                category_id=id_categoria_ia, exclude_report_id=report_id, days=15
            )

            # Verificación por vecindad geográfica
            es_duplicado_geografico = self.clustering_service.is_duplicate(report, historical_reports)
            es_duplicado_real = False
            max_similitud = 0.0
            id_reporte_duplicado = None
            
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
                        
                        if similitud > max_similitud:
                            max_similitud = similitud
                            
                        if similitud >= 0.75:
                            logger.warning(f"CONFIRMADO: Duplicado real con reporte #{reporte_conflicto['id']}.")
                            es_duplicado_real = True
                            id_reporte_duplicado = reporte_conflicto['id']
                            observacion_final = f"Reporte duplicado. (Cercanía espacial + Similitud visual con #{reporte_conflicto['id']}: {similitud*100:.1f}%)."
                            break 
                            
                    except Exception as e:
                        logger.warning(f"Error al comparar con reporte #{reporte_conflicto['id']}: {e}")
                        continue 

            # Registrar telemetria en JSONL local
            self._log_telemetry({
                "report_id": report_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_category_id": report['categoryId'],
                "original_category_name": nombre_categoria_original,
                "final_category_assigned": categoria_ia,
                "category_was_changed": categoria_modificada,
                "text_model": text_decision,
                "vision_model": clip_result,
                "fusion_result": fusion_result,
                "duplicate_analysis": {
                    "spatial_proximity_found": es_duplicado_geografico,
                    "is_duplicate": es_duplicado_real,
                    "max_visual_similarity": max_similitud,
                    "duplicate_of_id": id_reporte_duplicado
                },
                "final_state": "DUPLICADO" if es_duplicado_real else "VALIDADO"
            })

            # 8. Guardar Evento en el historial y Notificar
            if es_duplicado_real:
                await self.repo.add_history_entry_and_notify(
                    report_id, id_duplicado, REPORT_STATES.get("DUPLICADO", "DUPLICADO"), observacion_final
                )
                logger.warning("Guardado como DUPLICADO")
            else:
                await self.repo.add_history_entry_and_notify(
                    report_id, id_validado, REPORT_STATES.get("VALIDADO", "VALIDADO"), observacion_final
                )
                logger.info("Guardado como VALIDADO")

        except Exception as item_error:
            logger.error(f"Error al procesar el reporte #{report_id}: {item_error}", exc_info=True)