FROM python:3.11-slim

# Evita que Python escriba archivos .pyc en el disco (ahorra espacio en el contenedor)
ENV PYTHONDONTWRITEBYTECODE=1

# Fuerza a Python a enviar los logs a la terminal en tiempo real (crítico para ver 'docker logs')
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -u 5678 -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "worker.py"]