# app/services/clustering_service.py
import numpy as np
from sklearn.cluster import DBSCAN

class ClusteringService:
    def __init__(self, eps_meters=40):
        # 40 metros de radio
        self.eps_meters = eps_meters
        self.earth_radius_meters = 6371000.0
        self.eps_radians = self.eps_meters / self.earth_radius_meters

    def is_duplicate(self, target_report: dict, historical_reports: list) -> bool:
        # Si no hay reportes previos en esta categoría, es imposible que sea duplicado
        if not historical_reports:
            return False

        # 1. Preparamos las coordenadas: [[lat, lon], ...]
        coords = [[r['latitude'], r['longitude']] for r in historical_reports]
        
        # 2. Agregamos el reporte actual al final de la lista
        coords.append([target_report['latitude'], target_report['longitude']])

        # 3. Convertimos a radianes
        coords_np = np.radians(np.array(coords))

        # 4. Ejecutamos el modelo
        model = DBSCAN(
            eps=self.eps_radians, 
            min_samples=2, # Con 2 puntos ya es un grupo
            algorithm='ball_tree', 
            metric='haversine'
        )
        model.fit(coords_np)

        # 5. Analizamos el resultado
        target_label = model.labels_[-1]

        # DBSCAN asigna -1 a los puntos que están solos (ruido). 
        # Si tiene un número (0, 1, 2...), significa que cayó dentro del radio de otro reporte.
        return target_label != -1