import numpy as np
from sklearn.neighbors import BallTree

class ClusteringService:
    def __init__(self, eps_meters=40):
        # 40 metros de radio
        self.eps_meters = eps_meters
        self.earth_radius_meters = 6371000.0
        self.eps_radians = self.eps_meters / self.earth_radius_meters

    def is_duplicate(self, target_report: dict, historical_reports: list) -> bool:
        # Si no hay reportes previos en esta categoría, no puede ser duplicado
        if not historical_reports:
            return False

        # 1. Preparamos las coordenadas históricas en radianes
        coords_historical = np.radians(np.array([
            [r['latitude'], r['longitude']] for r in historical_reports
        ]))

        # 2. Construimos el árbol espacial solo con los datos históricos
        tree = BallTree(coords_historical, metric='haversine')

        # 3. Preparamos el reporte objetivo (nuevo) en radianes
        target_coords = np.radians(np.array([
            [target_report['latitude'], target_report['longitude']]
        ])).reshape(1, -1)

        # 4. Buscamos cuántos puntos históricos caen dentro del radio (eps_radians)
        indices = tree.query_radius(target_coords, r=self.eps_radians)

        # 5. Si la lista de índices encontrados tiene elementos, significa que hay duplicados
        return len(indices[0]) > 0