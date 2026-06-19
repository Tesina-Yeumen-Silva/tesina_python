import os
import json
import aio_pika
from datetime import datetime, timezone
from app.config.logger import logger

class RabbitMQManager:
    def __init__(self):
        # Configuración de infraestructura mediante variables de entorno
        self.url = os.getenv("RABBITMQ_URL", "amqp://admin:admin123@localhost")
        self.connection = None
        self.channel = None

    async def connect(self):
        """Establece una conexión robusta con RabbitMQ si no existe una activa."""
        if not self.connection or self.connection.is_closed:
            # connect_robust maneja automáticamente reconexiones si el servidor se cae temporalmente
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            logger.info("Conexión con RabbitMQ establecida.")

    async def close(self):
        """Cierra de forma limpia la conexión y los canales abiertos."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Desconectado de RabbitMQ.")

    async def publish_result(self, report_id: int, status: str):
        """Publica el resultado del análisis de IA en la cola de salida."""
        await self.connect()
        
        # Estructura del evento/mensaje a enviar al backend principal
        message = {
            "reportId": report_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        routing_key_results = os.getenv("RABBITMQ_QUEUE_RESULTS", "reports.results")
        
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=routing_key_results
        )
        logger.info(f"Resultado de reporte {report_id} publicado en RabbitMQ en la cola '{routing_key_results}' con estado {status}")

rabbitmq_manager = RabbitMQManager()