from sentence_transformers import SentenceTransformer, util
import torch

class CategoryClassifierService:
    def __init__(self):
        self.model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

        self.semantic_map = {
            "Acequias y Drenajes": [
                "acequia tapada, obstruida o con agua estancada",
                "desagüe bloqueado o sin funcionar en la calle",
                "inundación por falla en el sistema de drenaje",
                "canal de riego roto, desbordado o con residuos",
                "agua estancada en la calle por falta de desagüe",
            ],
            "Alumbrado Público": [
                "luminaria apagada, farola rota o falta de alumbrado público",
                "poste de luz caído o lámpara sin funcionar en la calle",
                "zona oscura de noche por falla en el alumbrado",
                "luz de la calle no enciende o parpadea constantemente",
                "cable eléctrico de alumbrado caído o pelado",
            ],
            "Arbolado Público": [
                "ramas caídas o árbol con riesgo de caída en la vía pública",
                "raíces que levantan la vereda o el pavimento",
                "necesidad de poda o tala de árbol peligroso en la calle",
                "árbol seco, inclinado o que bloquea el paso peatonal",
                "árbol caído sobre la calzada o vereda",
            ],
            "Baches y Pavimentación": [
                "pozo, bache o hundimiento en el asfalto o calzada",
                "hoyo en la calle que daña los autos o las ruedas",
                "pavimento roto, agrietado o destruido en la vía",
                "bache profundo que rompe cubiertas o llantas",
                "asfalto en mal estado con pozos o depresiones graves",
            ],
            "Limpieza y Residuos": [
                "basura acumulada o residuos en la vía pública",
                "escombros, desechos o falta de barrido en la calle",
                "contenedor desbordado o bolsas de basura abandonadas",
                "microbasural o residuos voluminosos en la vereda",
                "suciedad o residuos domiciliarios en espacio público",
            ],
            "Plazas y Parques": [
                "banco roto, juego dañado o infraestructura deteriorada en plaza",
                "plaza o parque con basura, maleza o en mal estado",
                "luminaria apagada o sendero deteriorado en espacio verde",
                "vandalismo o grafiti en mobiliario de plaza o parque",
                "pasto sin cortar o árboles sin mantenimiento en parque público",
            ],
            "Semáforos y Señalización": [
                "semáforo apagado, roto o con luz intermitente",
                "señal de tránsito caída, girada o ilegible",
                "cartel vial dañado, vandalizado o faltante",
                "semáforo peatonal sin funcionar o con tiempos incorrectos",
                "demarcación vial borrada o en mal estado en la calzada",
            ],
            "Veredas y Accesibilidad": [
                "vereda rota, levantada o con baldosas faltantes",
                "obstáculo en la vereda que impide el paso peatonal",
                "rampa de accesibilidad dañada o inexistente en esquina",
                "vereda intransitable por obras, raíces o material abandonado",
                "falta de rampa o barrera arquitectónica para personas con movilidad reducida",
            ],
            "Agua y Cloacas": [
                "pérdida de agua, caño roto o agua brotando en la calle",
                "cloaca desbordada, tapada o con mal olor en la vía pública",
                "boca de acceso cloacal rota, faltante o sin tapa",
                "charco permanente por pérdida de red de agua",
                "rotura de caño de agua potable en la calzada o vereda",
            ],
        }

        self._build_index()

    def _build_index(self):
        self._index = {}
        for category, phrases in self.semantic_map.items():
            self._index[category] = self.model.encode(
                phrases,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

    def classify_text(self, description: str, categories: list[str]) -> dict:
        query_embedding = self.model.encode(
            description,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = {}
        for category in categories:
            if category not in self._index:
                continue
            sims = util.dot_score(query_embedding, self._index[category])[0]
            top_k = torch.topk(sims, k=min(2, len(sims))).values
            scores[category] = float(top_k.mean())

        return scores