import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.config.logger import logger

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL no está configurada en las variables de entorno.")

if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Eliminar el parámetro 'schema' de la URL ya que asyncpg no lo admite
if "schema=" in database_url:
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    u = urlparse(database_url)
    q = parse_qsl(u.query)
    q = [(k, v) for k, v in q if k != 'schema']
    database_url = urlunparse(u._replace(query=urlencode(q)))

engine = create_async_engine(
    database_url,
    echo=False,
    pool_size=10,
    max_overflow=20
)

async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

class Database:
    def __init__(self):
        self.session_factory = async_session

    async def connect(self):
        logger.info("Conexión con PostgreSQL (SQLAlchemy) establecida de forma asíncrona.")

    async def disconnect(self):
        await engine.dispose()
        logger.info("Conexión con PostgreSQL cerrada.")

    def _convert_params(self, query: str, args: tuple) -> tuple[str, dict]:
        """Convierte los placeholders estilo $1, $2... al formato :p1, :p2... de SQLAlchemy."""
        sql, params = query, {}
        for idx, arg in enumerate(args):
            sql = sql.replace(f"${idx + 1}", f":p{idx + 1}")
            params[f"p{idx + 1}"] = arg
        return sql, params

    async def query_raw(self, query: str, *args):
        sql, params = self._convert_params(query, args)
        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            return [dict(row._mapping) for row in result.fetchall()]

    async def execute_raw(self, query: str, *args):
        sql, params = self._convert_params(query, args)
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(text(sql), params)

db = Database()

async def connect_db():
    await db.connect()

async def disconnect_db():
    await db.disconnect()
