# app/repositories/report_repository.py
import json

from app.config.prisma_db import db
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

    async def get_recent_reports_by_category(self, category_id: int, exclude_report_id: int, days: int = 15) -> list:
        """
        Busca reportes activos de la misma categoría de los últimos X días.
        Excluye resueltos, rechazados, duplicados y reportes borrados lógicamente.
        """
        date_threshold = datetime.utcnow() - timedelta(days=days)
        date_threshold_str = date_threshold.isoformat()

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
              AND r."createdAt" >= CAST($3 AS TIMESTAMP)
              AND ue.rn = 1
              AND ue.estado_actual NOT IN ('Resuelto', 'Rechazado', 'Duplicado')
              AND r."deletedAt" IS NULL 
        """
        return await db.query_raw(sql_query, category_id, exclude_report_id, date_threshold_str)

    async def get_state_id_by_name(self, state_name: str) -> int:
        state = await db.reportstate.find_unique(where={"name": state_name})
        if not state:
            raise ValueError(f"El estado '{state_name}' no existe en la base de datos.")
        return state.id

    async def get_category_id_by_name(self, category_name: str) -> int:
        category = await db.reportcategory.find_unique(where={"name": category_name})
        if not category:
            raise ValueError(f"La categoría '{category_name}' no existe en la base de datos.")
        return category.id

    async def update_report_category(self, report_id: int, category_id: int):
        await db.report.update(
            where={"id": report_id},
            data={"categoryId": category_id}
        )

    async def add_history_entry(self, report_id: int, state_id: int, observation: str):
        await db.reporthistory.create(
            data={
                "reportId": report_id,
                "stateId": state_id,
                "observation": observation
            }
        )

    async def add_history_entry_and_notify(self, report_id: int, state_id: int, state_name: str, observation: str):
        """
        Guarda el evento en el historial y notifica instantáneamente a Node.js
        """
        # 1. Guardar en la base de datos
        await db.reporthistory.create(
            data={
                "reportId": report_id,
                "stateId": state_id,
                "observation": observation
            }
        )

        # 2. Formato de los datos
        payload = {
            "reportId": report_id,
            "nuevoEstado": state_name,
            "mensaje": observation
        }
        payload_json = json.dumps(payload)

        # 3. Disparar el evento de forma SEGURA.
        await db.query_raw('SELECT pg_notify($1, $2)::text', 'report_updates', payload_json)
        
        logger.info(f"Notificación enviada a Node -> Reporte {report_id}: {state_name}")