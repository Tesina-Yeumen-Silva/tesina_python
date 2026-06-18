# app/repositories/report_repository.py
import json

from app.config.db import db
from datetime import datetime, timedelta, timezone
from app.config.logger import logger

class ReportRepository:
    
    async def get_pending_reports(self, pendiente_state_name: str) -> list:
        """
        Busca todos los reportes cuyo ÚLTIMO estado registrado en el historial 
        sea estrictamente 'Pendiente' y que NO hayan sido borrados lógicamente.
        """
        sql_query = """
            WITH UltimoEstado AS (
                SELECT rh."reportId", rs.name as estado_actual,
                       ROW_NUMBER() OVER(PARTITION BY rh."reportId" ORDER BY rh."createdAt" DESC) as rn
                FROM "ReportHistory" rh
                JOIN "ReportState" rs ON rh."stateId" = rs.id
                WHERE rs."deletedAt" IS NULL
            )
            SELECT r.id, r.address, r.latitude, r.longitude, r.description, r."imageUrl", r."categoryId", r."userId"
            FROM "Report" r
            JOIN UltimoEstado ue ON r.id = ue."reportId"
            WHERE ue.rn = 1 
              AND ue.estado_actual = $1
              AND r."deletedAt" IS NULL 
        """
        return await db.query_raw(sql_query, pendiente_state_name)

    async def get_report_by_id(self, report_id: int) -> dict | None:
        """
        Busca un reporte específico por su ID.
        """
        sql_query = """
            SELECT id, address, latitude, longitude, description, "imageUrl", "categoryId", "userId"
            FROM "Report"
            WHERE id = $1 AND "deletedAt" IS NULL
        """
        results = await db.query_raw(sql_query, report_id)
        return results[0] if results else None

    async def get_recent_reports_by_category(self, category_id: int, exclude_report_id: int, days: int = 15) -> list:
        """
        Busca reportes activos de la misma categoría de los últimos X días.
        Excluye resueltos, rechazados, duplicados y reportes borrados lógicamente.
        """
        date_threshold = datetime.utcnow() - timedelta(days=days)

        sql_query = """
            WITH UltimoEstado AS (
                SELECT rh."reportId", rs.name as estado_actual,
                       ROW_NUMBER() OVER(PARTITION BY rh."reportId" ORDER BY rh."createdAt" DESC) as rn
                FROM "ReportHistory" rh
                JOIN "ReportState" rs ON rh."stateId" = rs.id
                WHERE rs."deletedAt" IS NULL
            )
            SELECT r.id, r.latitude, r.longitude, r."imageUrl" 
            FROM "Report" r
            JOIN UltimoEstado ue ON r.id = ue."reportId"
            WHERE r."categoryId" = $1
              AND r.id != $2
              AND r."createdAt" >= $3
              AND ue.rn = 1
              AND ue.estado_actual NOT IN ('Resuelto', 'Rechazado', 'Duplicado')
              AND r."deletedAt" IS NULL 
        """
        return await db.query_raw(sql_query, category_id, exclude_report_id, date_threshold)

    async def get_state_id_by_name(self, state_name: str) -> int:
        sql = 'SELECT id FROM "ReportState" WHERE name = $1 AND "deletedAt" IS NULL'
        results = await db.query_raw(sql, state_name)
        if not results:
            raise ValueError(f"El estado '{state_name}' no existe en la base de datos.")
        return results[0]["id"]

    async def get_category_id_by_name(self, category_name: str) -> int:
        sql = 'SELECT id FROM "ReportCategory" WHERE name = $1 AND "deletedAt" IS NULL'
        results = await db.query_raw(sql, category_name)
        if not results:
            raise ValueError(f"La categoría '{category_name}' no existe en la base de datos.")
        return results[0]["id"]

    async def update_report_category(self, report_id: int, category_id: int):
        sql = 'UPDATE "Report" SET "categoryId" = $2 WHERE id = $1'
        await db.execute_raw(sql, report_id, category_id)

    async def add_history_entry(self, report_id: int, state_id: int, observation: str):
        sql = 'INSERT INTO "ReportHistory" ("reportId", "stateId", "observation", "createdAt") VALUES ($1, $2, $3, NOW())'
        await db.execute_raw(sql, report_id, state_id, observation)

    async def add_history_entry_and_notify(self, report_id: int, state_id: int, state_name: str, observation: str):
        """
        Guarda el evento en el historial y notifica instantáneamente a Node.js a través de RabbitMQ
        """
        # 1. Guardar en la base de datos
        sql = 'INSERT INTO "ReportHistory" ("reportId", "stateId", "observation", "createdAt") VALUES ($1, $2, $3, NOW())'
        await db.execute_raw(sql, report_id, state_id, observation)

        # 2. Notificar a Node.js a través de RabbitMQ
        try:
            from app.config.rabbitmq import rabbitmq_manager
            await rabbitmq_manager.publish_result(report_id, state_name)
        except Exception as e:
            logger.error(f"Error al enviar notificación a RabbitMQ para el reporte {report_id}: {e}", exc_info=True)