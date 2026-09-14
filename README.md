# Motor de IA — Mendoza Reporta

Worker asíncrono que procesa reportes ciudadanos con IA (CLIP + MPNet) consumiendo mensajes de RabbitMQ.

## Arquitectura

```
worker.py                    ← Punto de entrada. Consume la cola RabbitMQ.
app/
├── config/
│   ├── db.py                ← Conexión async a PostgreSQL (SQLAlchemy + asyncpg)
│   ├── rabbitmq.py          ← Conexión robusta a RabbitMQ (aio-pika)
│   └── logger.py            ← Logger con rotación de archivos
├── repositories/
│   └── report_repository.py ← Consultas SQL sobre los reportes
├── services/
│   ├── clip_services.py     ← Clasificación y comparación de imágenes (CLIP)
│   ├── category_services.py ← Clasificación semántica de texto (MPNet)
│   ├── clustering_service.py← Detección de duplicados geográficos (BallTree)
│   └── report_decision_service.py ← Fusión de decisiones texto/imagen
└── use_cases/
    └── process_pending_reports.py ← Orquestador del pipeline de IA
```

## Flujo de procesamiento

1. RabbitMQ entrega un mensaje con `{ "action": "validate_report", "reportId": N }`
2. Se descarga la imagen del reporte
3. **CLIP** verifica que sea foto real → exterior urbano → problema detectado
4. **MPNet** clasifica semánticamente la descripción del usuario
5. Se fusionan ambas decisiones para asignar la categoría final
6. **BallTree** detecta si hay reportes similares en un radio de 40m (últimos 15 días)
7. Si hay cercanía, CLIP compara visualmente las imágenes (umbral: 75% similitud)
8. El resultado (`Validado` / `Rechazado` / `Duplicado`) se guarda en DB y se notifica por RabbitMQ

## Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `DATABASE_URL` | URL de PostgreSQL | *(requerida)* |
| `RABBITMQ_URL` | URL de RabbitMQ | `amqp://admin:admin123@localhost` |
| `RABBITMQ_QUEUE_VALIDATE` | Cola de entrada | `reports.validate` |
| `RABBITMQ_QUEUE_RESULTS` | Cola de salida | `reports.results` |
| `CLIP_REAL_PHOTO_THRESHOLD` | Umbral foto real | `0.65` |
| `CLIP_OUTDOOR_THRESHOLD` | Umbral exterior urbano | `0.60` |
| `CLIP_PROBLEM_THRESHOLD` | Umbral problema detectado | `0.40` |

## Levantar con Docker

```bash
docker build -t mendoza-reporta-ia .
docker run --env-file .env mendoza-reporta-ia
```

## Desarrollo local

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python worker.py
```
