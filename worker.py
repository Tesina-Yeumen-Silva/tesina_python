# worker.py
import asyncio
import json
import aio_pika
from app.config.db import connect_db, disconnect_db
from app.config.rabbitmq import rabbitmq_manager
from app.repositories.report_repository import ReportRepository
from app.services.clip_services import ClipService
from app.services.category_services import CategoryClassifierService
from app.services.report_decision_service import ReportDecisionService
from app.use_cases.process_pending_reports import ProcessPendingReportsUseCase
from app.config.logger import logger
from app.services.clustering_service import ClusteringService 

async def main():
    logger.info("=======================================================")
    logger.info("INICIANDO MOTOR ASÍNCRONO DE IA CON RABBITMQ: MENDOZA REPORTA")
    logger.info("=======================================================")
    
    await connect_db()
    await rabbitmq_manager.connect()

    clip_service = ClipService()
    text_service = CategoryClassifierService()
    decision_service = ReportDecisionService(text_threshold=0.45)
    clustering_service = ClusteringService() 
    
    repo = ReportRepository()
    
    use_case = ProcessPendingReportsUseCase(
        repo, clip_service, text_service, decision_service , clustering_service
    )

    # Declarar e iniciar escucha en la cola
    queue_name = "reports.validate"
    queue = await rabbitmq_manager.channel.declare_queue(queue_name, durable=True)
    await rabbitmq_manager.channel.set_qos(prefetch_count=1)

    logger.info(f"Sistema listo y escuchando reportes entrantes en la cola '{queue_name}'...")

    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                report_id = body.get("reportId")
                action = body.get("action")
                
                logger.info(f"Mensaje recibido de RabbitMQ: {body}")
                
                if action == "validate_report" and report_id is not None:
                    await use_case.execute(report_id)
                else:
                    logger.warning(f"Acción o ID de reporte no válido recibido: {body}")
            except Exception as e:
                logger.error(f"Error procesando mensaje de la cola: {e}", exc_info=True)

    try:
        # Comenzar a consumir
        await queue.consume(on_message)
        
        # Mantener el proceso vivo indefinidamente
        while True:
            await asyncio.sleep(3600)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Deteniendo servicio de IA de forma segura (interrupción manual)...")
    finally:
        await rabbitmq_manager.close()
        await disconnect_db()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servicio detenido.")