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
                "cuneta tapada, con hojas, ramas o basura",
                "zanjón desbordado o sucio en la calle",
                "acequia sucia con barro y botellas",
                "puente de acequia tapado que desborda agua",
                "alcantarilla de calle tapada de hojas",
                "acequia tapada que cuando llueve se inunda todo",
                "inundación de acequia por lluvia y basura",
                "acequia tapada de basura se inunda todo",
                "acequia que se desborda con la lluvia",
                "acequia tapada",
                "desagüe obstruido",
                "acequia",
                "zanjón sucio",
                "cuneta",
                "desborde de acequia",
            ],
            "Alumbrado Público": [
                "luminaria apagada, farola rota o falta de alumbrado público",
                "poste de luz caído o lámpara sin funcionar en la calle",
                "zona oscura de noche por falla en el alumbrado",
                "luz de la calle no enciende o parpadea constantemente",
                "cable eléctrico de alumbrado caído o pelado",
                "farola rota o apagada en la cuadra",
                "luminaria de alumbrado que parpadea o no enciende",
                "poste de luz quebrado, caído o inclinado con peligro de cables",
                "cables de luz eléctrica colgando en la calle",
                "foco de luz quemado en la calle",
                "calle a oscuras sin luz",
                "luz apagada",
                "farola rota",
                "poste de luz caído",
                "sin luz en la calle",
                "farola",
            ],
            "Arbolado Público": [
                "ramas caídas o árbol con riesgo de caída en la vía pública",
                "raíces que levantan la vereda o el pavimento",
                "necesidad de poda o tala de árbol peligroso en la calle",
                "árbol seco, inclinado o que bloquea el paso peatonal",
                "árbol caído sobre la calzada o vereda",
                "árbol caído por el viento zonda sobre la calle o vereda",
                "rama de árbol que se quebró o está por caerse",
                "ramas grandes que tocan los cables de luz y hacen chispas",
                "árbol seco o en peligro de derrumbe",
                "plátano o árbol que necesita poda urgente",
                "árbol caído",
                "rama caída",
                "poda de árbol",
                "árbol peligroso",
                "árbol zonda",
                "rama caída calle",
            ],
            "Baches y Pavimentación": [
                "pozo, bache o hundimiento en el asfalto o calzada",
                "hoyo en la calle que daña los autos o las ruedas",
                "pavimento roto, agrietado o destruido en la vía",
                "bache profundo que rompe cubiertas o llantas",
                "asfalto en mal estado con pozos o depresiones graves",
                "bache enorme en el pavimento o asfalto roto",
                "cráter o pozo en la calle que rompe ruedas",
                "hundimiento del asfalto o calzada deformada",
                "bache profundo y peligroso en la avenida",
                "calle de tierra destruida o asfalto agrietado",
                "pozo grande",
                "pozo en la calle",
                "bache o pozo",
                "pozo en el asfalto",
                "calle rota",
                "bache grande",
                "bacheazo",
                "cráter en la calle",
                "pozo asfalto",
            ],
            "Limpieza y Residuos": [
                "basura acumulada o residuos en la vía pública",
                "escombros, desechos o falta de barrido en la calle",
                "contenedor desbordado o bolsas de basura abandonadas",
                "microbasural o residuos voluminosos en la vereda",
                "suciedad o residuos domiciliarios en espacio público",
                "microbasural clandestino en la esquina o vereda",
                "montículo de escombros, tierra o restos de poda abandonados",
                "contenedor de basura desbordado de residuos",
                "bolsas de basura rotas tiradas en la calle con mal olor",
                "acumulación de mugre o desperdicios en vía pública",
                "basura acumulada",
                "microbasural",
                "contenedor lleno",
                "basura en la calle",
                "basurero lleno",
                "escombros en la vereda",
            ],
            "Plazas y Parques": [
                "banco roto, juego dañado o infraestructura deteriorada en plaza",
                "plaza o parque con basura, maleza o en mal estado",
                "luminaria apagada o sendero deteriorado en espacio verde",
                "vandalismo o grafiti en mobiliario de plaza o parque",
                "pasto sin cortar o árboles sin mantenimiento en parque público",
                "juegos rotos, oxidados o peligrosos en la plaza de los niños",
                "banco de plaza partido, roto o pintado con grafiti",
                "pasto altísimo sin cortar y malezas en el parque o espacio verde",
                "hamaca o tobogán roto en plazoleta infantil",
                "plaza abandonada y descuidada",
                "pasto alto",
                "plaza rota",
                "juegos rotos",
                "mantenimiento de plaza",
                "banco de parque roto",
            ],
            "Semáforos y Señalización": [
                "semáforo apagado, roto o con luz intermitente",
                "señal de tránsito caída, girada o ilegible",
                "cartel vial dañado, vandalizado o faltante",
                "semáforo peatonal sin funcionar o con tiempos incorrectos",
                "demarcación vial borrada o en mal estado en la calzada",
                "semáforo roto, caído o titilando en amarillo / intermitente",
                "cartel de señalización vial o de contramano doblado o en el piso",
                "cartel de pare o ceda el paso tirado o vandalizado",
                "semáforo vehicular trabado en rojo o apagado",
                "semáforo peatonal roto",
                "semáforo roto",
                "semáforo apagado",
                "cartel de calle roto",
                "señal de tránsito",
                "cartel de calle caído",
            ],
            "Veredas y Accesibilidad": [
                "vereda rota, levantada o con baldosas faltantes",
                "obstáculo en la vereda que impide el paso peatonal",
                "rampa de accesibilidad dañada o inexistente en esquina",
                "vereda intransitable por obras, raíces o material abandonado",
                "falta de rampa o barrera arquitectónica para personas con movilidad reducida",
                "baldosas flojas, sueltas o levantadas en la vereda",
                "vereda rota intransitable por raíces de árbol",
                "rampa de discapacitados rota, empinada o bloqueada en la esquina",
                "pozo o trampa en la vereda donde tropiezan los peatones",
                "falta de accesibilidad para sillas de ruedas o cochecitos",
                "vereda rota",
                "baldosas flojas",
                "rampa de discapacitados rota",
                "vereda destruida",
                "rampa rota",
                "vereda con baldosas rotas",
            ],
            "Agua y Cloacas": [
                "pérdida de agua, caño roto o agua brotando en la calle",
                "cloaca desbordada, tapada o con mal olor en la vía pública",
                "boca de acceso cloacal rota, faltante o sin tapa",
                "charco permanente por pérdida de red de agua",
                "rotura de caño de agua potable en la calzada o vereda",
                "pérdida de agua potable brotando del pavimento o vereda",
                "caño maestro roto que inunda la calle con agua limpia",
                "cloaca rebasada o tapada con aguas servidas y olor a podrido",
                "tapa de alcantarilla o boca de registro cloacal rota o sin tapa",
                "brote de agua cloacal en la calle",
                "caño roto",
                "pérdida de agua",
                "cloaca tapada",
                "tapa de cloaca rota",
                "caño maestro roto",
                "desborde cloacal",
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

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza texto coloquial, modismos de chat y errores ortográficos comunes
        en reportes ciudadanos antes de la vectorización con SentenceTransformers.
        """
        if not text:
            return ""
            
        import re
        t = text.lower()

        # 1. Colapsar repetición excesiva de caracteres (ej: "rrttto" -> "rto", "lllenooo" -> "lleno")
        t = re.sub(r'([a-zA-Z])\1{2,}', r'\1', t)
        
        # 2. Correcciones ortográficas y abreviaturas frecuentes de mensajería (SMS/WhatsApp/Redes)
        replacements = [
            (r'\bq\b', 'que'),
            (r'\bx\b', 'por'),
            (r'\bxq\b', 'porque'),
            (r'\bporq\b', 'porque'),
            (r'\bd\b', 'de'),
            (r'\bc\b', 'se'),
            (r'\btan\b', 'estan'),
            (r'\btda\b', 'toda'),
            (r'\btdo\b', 'todo'),
            (r'\btdos\b', 'todos'),
            (r'\btdas\b', 'todas'),
            (r'\bksa\b', 'casa'),
            (r'\bcs\b', 'casa'),
            (r'\bcaio\b', 'cayo'),
            (r'\bcayo\b', 'cayó'),
            (r'\bai\b', 'hay'),
            (r'\bay\b', 'hay'),
            (r'\bholor\b', 'olor'),
            (r'\bawa\b', 'agua'),
            (r'\bahua\b', 'agua'),
            (r'\bsho\b', 'yo'),
            (r'\bshno\b', 'lleno'),
            (r'\bshueve\b', 'llueve'),
            (r'\byueve\b', 'llueve'),
            (r'\basra\b', 'basura'),
            (r'\bbasra\b', 'basura'),
            (r'\bvachhe\b', 'bache'),
            (r'\bvache\b', 'bache'),
            (r'\bianta\b', 'llanta'),
            (r'\baoto\b', 'auto'),
            (r'\balluda\b', 'ayuda'),
            (r'\bayudaa\b', 'ayuda'),
            (r'\bezcombro\b', 'escombro'),
            (r'\becsombro\b', 'escombro'),
            (r'\brratas\b', 'ratas'),
            (r'\brraises\b', 'raices'),
            (r'\brraices\b', 'raíces'),
            (r'\brrompieron\b', 'rompieron'),
            (r'\brronpieron\b', 'rompieron'),
            (r'\brrompeeron\b', 'rompieron'),
            (r'\bplasita\b', 'placita'),
            (r'\bplazita\b', 'placita'),
            (r'\bamaca\b', 'hamaca'),
            (r'\bamacas\b', 'hamacas'),
            (r'\bzemaforo\b', 'semáforo'),
            (r'\bsemaforo\b', 'semáforo'),
            (r'\bkrusa\b', 'cruza'),
            (r'\bchokaar\b', 'chocar'),
            (r'\bchokar\b', 'chocar'),
            (r'\branpa\b', 'rampa'),
            (r'\bvvaldosas\b', 'baldosas'),
            (r'\bvaldosas\b', 'baldosas'),
            (r'\bcilla\b', 'silla'),
            (r'\bavuelo\b', 'abuelo'),
            (r'\btnancada\b', 'estancada'),
            (r'\bobvra\b', 'obra'),
            (r'\bdgaton\b', 'dejaron'),
            (r'\bklooaca\b', 'cloaca'),
            (r'\bkloaca\b', 'cloaca'),
            (r'\brevalsoo\b', 'rebalso'),
            (r'\brevalso\b', 'rebasó'),
            (r'\bprrdida\b', 'pérdida'),
        ]
        for pattern, repl in replacements:
            t = re.sub(pattern, repl, t)

        return t

    def classify_text(self, description: str) -> dict:
        """Compara la descripción con el índice y retorna los puntajes por categoría."""
        normalized_description = self.normalize_text(description)
        query_embedding = self.model.encode(normalized_description, convert_to_tensor=True, normalize_embeddings=True)
        
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