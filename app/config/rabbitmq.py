import aio_pika
import os
import json
from datetime import datetime, timezone
from app.config.logger import logger

class RabbitMQManager:
    def __init__(self):
        self.url = os.getenv("RABBITMQ_URL", "amqp://admin:admin123@localhost")
        self.connection = None
        self.channel = None

    async def connect(self):
        if not self.connection or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            logger.info("Conexión con RabbitMQ establecida.")

    async def close(self):
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Desconectado de RabbitMQ.")

    async def publish_result(self, report_id: int, status: str):
        await self.connect()
        message = {
            "reportId": report_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key="reports.results"
        )
        logger.info(f"Resultado de reporte {report_id} publicado en RabbitMQ con estado {status}")

rabbitmq_manager = RabbitMQManager()
