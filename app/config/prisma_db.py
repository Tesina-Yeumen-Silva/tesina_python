from prisma import Prisma
from app.config.logger import logger

# Instanciamos el cliente de Prisma global
db = Prisma(auto_register=True)

async def connect_db():
    if not db.is_connected():
        await db.connect()
        logger.info("Conexión con PostgreSQL establecida a través de Prisma.")

async def disconnect_db():
    if db.is_connected():
        await db.disconnect()
        logger.info("Desconectado de PostgreSQL.")