import torch
from sentence_transformers import SentenceTransformer, util

class CategoryClassifierService:
    def __init__(self, threshold: float = 0.4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es", device=self.device)
        self.threshold = threshold
        
        self.semantic_map = {
            "Acequias y Drenajes": [
                "acequia tapada, obstruida o con agua estancada",
                "desagüe bloqueado o sin funcionar en la calle",
                "inundación por falla en el sistema de drenaje",
                "canal de riego roto, desbordado o con residuos",
                "agua estancada en la calle por falta de desagüe",
                "acequia tapada",
                "desagüe obstruido",
                "acequia",
                "zanjón sucio",
            ],
            "Alumbrado Público": [
                "luminaria apagada, farola rota o falta de alumbrado público",
                "poste de luz caído o lámpara sin funcionar en la calle",
                "zona oscura de noche por falla en el alumbrado",
                "luz de la calle no enciende o parpadea constantemente",
                "cable eléctrico de alumbrado caído o pelado",
                "luz apagada",
                "farola rota",
                "poste de luz caído",
                "sin luz en la calle",
            ],
            "Arbolado Público": [
                "ramas caídas o árbol con riesgo de caída en la vía pública",
                "raíces que levantan la vereda o el pavimento",
                "necesidad de poda o tala de árbol peligroso en la calle",
                "árbol seco, inclinado o que bloquea el paso peatonal",
                "árbol caído sobre la calzada o vereda",
                "árbol caído",
                "rama caída",
                "poda de árbol",
                "árbol peligroso",
            ],
            "Baches y Pavimentación": [
                "pozo, bache o hundimiento en el asfalto o calzada",
                "hoyo en la calle que daña los autos o las ruedas",
                "pavimento roto, agrietado o destruido en la vía",
                "bache profundo que rompe cubiertas o llantas",
                "asfalto en mal estado con pozos o depresiones graves",
                "pozo grande",
                "pozo en la calle",
                "bache o pozo",
                "pozo en el asfalto",
                "calle rota",
                "bache grande",
            ],
            "Limpieza y Residuos": [
                "basura acumulada o residuos en la vía pública",
                "escombros, desechos o falta de barrido en la calle",
                "contenedor desbordado o bolsas de basura abandonadas",
                "microbasural o residuos voluminosos en la vereda",
                "suciedad o residuos domiciliarios en espacio público",
                "basura acumulada",
                "microbasural",
                "contenedor lleno",
                "basura en la calle",
            ],
            "Plazas y Parques": [
                "banco roto, juego dañado o infraestructura deteriorada en plaza",
                "plaza o parque con basura, maleza o en mal estado",
                "luminaria apagada o sendero deteriorado en espacio verde",
                "vandalismo o grafiti en mobiliario de plaza o parque",
                "pasto sin cortar o árboles sin mantenimiento en parque público",
                "pasto alto",
                "plaza rota",
                "juegos rotos",
                "mantenimiento de plaza",
            ],
            "Semáforos y Señalización": [
                "semáforo apagado, roto o con luz intermitente",
                "señal de tránsito caída, girada o ilegible",
                "cartel vial dañado, vandalizado o faltante",
                "semáforo peatonal sin funcionar o con tiempos incorrectos",
                "demarcación vial borrada o en mal estado en la calzada",
                "semáforo roto",
                "semáforo apagado",
                "cartel de calle roto",
                "señal de tránsito",
            ],
            "Veredas y Accesibilidad": [
                "vereda rota, levantada o con baldosas faltantes",
                "obstáculo en la vereda que impide el paso peatonal",
                "rampa de accesibilidad dañada o inexistente en esquina",
                "vereda intransitable por obras, raíces o material abandonado",
                "falta de rampa o barrera arquitectónica para personas con movilidad reducida",
                "vereda rota",
                "baldosas flojas",
                "rampa de discapacitados rota",
                "vereda destruida",
            ],
            "Agua y Cloacas": [
                "pérdida de agua, caño roto o agua brotando en la calle",
                "cloaca desbordada, tapada o con mal olor en la vía pública",
                "boca de acceso cloacal rota, faltante o sin tapa",
                "charco permanente por pérdida de red de agua",
                "rotura de caño de agua potable en la calzada o vereda",
                "caño roto",
                "pérdida de agua",
                "cloaca tapada",
                "tapa de cloaca rota",
            ],
        }
        
        self._build_index()

    def _build_index(self):
        """Convierte todas las frases en vectores para búsqueda rápida."""
        self.corpus_embeddings = []
        self.mapping = [] 
        
        for category, phrases in self.semantic_map.items():
            embs = self.model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True)
            self.corpus_embeddings.append(embs)
            self.mapping.extend([category] * len(phrases))
            
        self.corpus_embeddings = torch.cat(self.corpus_embeddings)

    def classify_text(self, description: str) -> dict:
        """Compara la descripción con el índice y retorna los puntajes por categoría."""
        query_embedding = self.model.encode(description, convert_to_tensor=True, normalize_embeddings=True)
        
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=10)[0]
        
        # Usamos el puntaje MÁXIMO por categoría para evitar penalizar por promedios
        scores = {cat: 0.0 for cat in self.semantic_map.keys()}
        for hit in hits:
            category = self.mapping[hit['corpus_id']]
            if hit['score'] > scores[category]:
                scores[category] = hit['score']
            
        return scores

    def get_best_category(self, description: str):
        """
        Retorna la mejor categoría si supera el umbral, o 'No identificada'.
        Método de utilidad para pruebas standalone. En el pipeline principal
        se usa classify_text() junto a ReportDecisionService.
        """
        scores = self.classify_text(description)
        best_cat = max(scores, key=scores.get)
        
        if scores[best_cat] < self.threshold:
            return "Categoría no identificada", scores[best_cat]
        return best_cat, scores[best_cat]