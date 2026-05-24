# worker.py
import asyncio
from app.config.prisma_db import connect_db, disconnect_db
from app.repositories.report_repository import ReportRepository
from app.services.clip_services import ClipService
from app.services.category_services import CategoryClassifierService
from app.services.report_decision_service import ReportDecisionService
from app.use_cases.process_pending_reports import ProcessPendingReportsUseCase

# 🚀 NUEVO: Importamos nuestro logger
from app.config.logger import logger
from python_backend.app.services.clustering_service import ClusteringService 

async def main():
    logger.info("=======================================================")
    logger.info("INICIANDO MOTOR ASÍNCRONO DE IA: MENDOZA REPORTA")
    logger.info("=======================================================")
    
    await connect_db()

    clip_service = ClipService()
    text_service = CategoryClassifierService()
    decision_service = ReportDecisionService(text_threshold=0.45)
    clustering_service = ClusteringService() 
    
    repo = ReportRepository()
    
    use_case = ProcessPendingReportsUseCase(
        repo, clip_service, text_service, decision_service , clustering_service
    )

    logger.info("Sistema listo y escuchando reportes entrantes...")

    try:
        while True:
            try:
                await use_case.execute()
            except Exception as e:
                logger.error(f"Fallo crítico en el ciclo de ejecución: {e}", exc_info=True)
            
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Deteniendo servicio de IA de forma segura (interrupción manual)...")
    finally:
        await disconnect_db()

if __name__ == "__main__":
    asyncio.run(main())