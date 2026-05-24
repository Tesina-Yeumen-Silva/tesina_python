import logging
import os
from logging.handlers import RotatingFileHandler

# Crear la carpeta de logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Archivo que rota automáticamente (guarda hasta 5 archivos de 5MB cada uno)
file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'worker_ia.log'), 
    maxBytes=5*1024*1024, 
    backupCount=5
)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("MendozaReportaIA")
logger.setLevel(logging.INFO) 
logger.addHandler(file_handler)
logger.addHandler(console_handler)