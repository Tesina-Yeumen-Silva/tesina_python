# Mendoza Reporta - AI Worker (Python)

Este repositorio contiene el subsistema de Inteligencia Artificial de **Mendoza Reporta**. Actúa como un worker asíncrono diseñado para auditar, clasificar y validar automáticamente los reportes urbanos generados por los ciudadanos.

## 🧠 Arquitectura Multimodal y Modelos

El worker implementa un pipeline de validación utilizando dos aproximaciones en paralelo, para luego aplicar un motor de decisión unificado (Soft Gating y Fusión Ponderada).

1. **Modelo de Visión (OpenAI CLIP ViT-B/32)**:
   - Extrae el vector de características de la imagen reportada.
   - Analiza si es una fotografía real y si pertenece a un entorno urbano exterior.
   - Clasifica el tipo de incidente visible (e.g. Baches, Arbolado, Acequias).

2. **Modelo de Texto (`hiiamsid/sentence_similarity_spanish_es` / MPNet-BETO)**:
   - Transforma la descripción textual del ciudadano en embeddings.
   - Mide la similitud de coseno contra un índice semántico estático para identificar la categoría de infraestructura urbana afectada.

3. **Clustering y Control Espacial (`BallTree`)**:
   - Detecta duplicidad analizando la distancia geoespacial entre reportes recientes de la misma categoría. En caso de cercanía, realiza una verificación de similitud visual directa (peritaje iterativo).

## 📋 Requisitos Previos

- **Python** 3.10 o superior.
- **Base de Datos**: Conexión a la instancia principal PostgreSQL del ecosistema.
- Hardware: Aunque se ejecuta eficientemente en CPU gracias a las versiones base de los modelos, el soporte CUDA se activa automáticamente si está disponible en el entorno.

## ⚙️ Configuración del Entorno

1. Copiar el archivo de variables de entorno:
   ```bash
   cp .env.example .env
   ```
2. Completar la variable `DATABASE_URL` para permitir la lectura y actualización de los reportes. Opcionalmente, se pueden ajustar los umbrales de validación (`CLIP_REAL_PHOTO_THRESHOLD`, etc).

## 🛠️ Instalación y Uso Local

1. Crear un entorno virtual e instalar las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Ejecutar el worker:
   ```bash
   python worker.py
   ```

El worker comenzará a observar la base de datos de manera periódica, descargando en paralelo las imágenes requeridas y anexando los registros de telemetría de IA de forma estructurada.

## 📊 Telemetría y Logs

Para entornos de auditoría o pruebas piloto, el sistema registra una traza completa de las inferencias (score de confianza, categoría corregida, consenso) en el archivo de registro `app/logs/ai_telemetry.jsonl`.
