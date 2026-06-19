from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import logging

logger = logging.getLogger("MendozaReportaIA")


class ClipService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ClipService iniciado en dispositivo: {self.device}")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.real_photo_labels = [
            "a real photograph taken with a camera outdoors",
            "a digital image, meme, screenshot, cartoon, drawing or AI generated image",
        ]
        self.outdoor_labels = [
            "an outdoor urban street scene with roads, sidewalks or public infrastructure",
            "an indoor scene, a person, a pet, food or a natural landscape",
        ]
        self.problem_labels = [
            "a blocked or flooded drainage ditch or canal on the street",       # Acequias y Drenajes
            "a broken or unlit streetlight or fallen electric pole",             # Alumbrado Público
            "a fallen tree, dangerous branches or roots lifting the sidewalk",   # Arbolado Público
            "a pothole, damaged pavement or broken road surface",                # Baches y Pavimentación
            "garbage, waste, rubble or trash accumulated on the street",         # Limpieza y Residuos
            "a damaged bench, broken playground or neglected public park",       # Plazas y Parques
            "a broken traffic light, fallen road sign or faded road markings",   # Semáforos y Señalización
            "a broken sidewalk, missing tiles or blocked pedestrian access",     # Veredas y Accesibilidad
            "a water leak, broken pipe or overflowing sewer on the street",     # Agua y Cloacas
            "a normal street or public space in good condition with no issues",  # descarte
            "an unrelated scene with no urban infrastructure problems visible",  # descarte
        ]
        self.label_to_category = {
            "a blocked or flooded drainage ditch or canal on the street":       "Acequias y Drenajes",
            "a broken or unlit streetlight or fallen electric pole":             "Alumbrado Público",
            "a fallen tree, dangerous branches or roots lifting the sidewalk":   "Arbolado Público",
            "a pothole, damaged pavement or broken road surface":                "Baches y Pavimentación",
            "garbage, waste, rubble or trash accumulated on the street":         "Limpieza y Residuos",
            "a damaged bench, broken playground or neglected public park":       "Plazas y Parques",
            "a broken traffic light, fallen road sign or faded road markings":   "Semáforos y Señalización",
            "a broken sidewalk, missing tiles or blocked pedestrian access":     "Veredas y Accesibilidad",
            "a water leak, broken pipe or overflowing sewer on the street":     "Agua y Cloacas",
        }
        self.no_problem_labels = {
            "a normal street or public space in good condition with no issues",
            "an unrelated scene with no urban infrastructure problems visible",
        }

        self.REAL_PHOTO_THRESHOLD = float(os.getenv("CLIP_REAL_PHOTO_THRESHOLD", 0.65))
        self.OUTDOOR_THRESHOLD    = float(os.getenv("CLIP_OUTDOOR_THRESHOLD",    0.60))
        self.PROBLEM_THRESHOLD    = float(os.getenv("CLIP_PROBLEM_THRESHOLD",    0.40))

        self._real_photo_text = self._tokenize_labels(self.real_photo_labels)
        self._outdoor_text    = self._tokenize_labels(self.outdoor_labels)
        self._problem_text    = self._tokenize_labels(self.problem_labels)

    def _tokenize_labels(self, labels: list[str]) -> dict:
        """Tokeniza una lista de etiquetas y mueve los tensores al device."""
        inputs = self.processor(text=labels, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_probs(self, image: Image.Image, labels: list[str], precomputed_text: dict) -> dict:
        """
        Calcula las probabilidades de cada etiqueta para una imagen dada.
        Reutiliza los tensores de texto pre-tokenizados para mayor eficiencia.
        """
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        inputs = {**image_inputs, **precomputed_text}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]
        return dict(zip(labels, probs.tolist()))

    def classify_image(self, image: Image.Image) -> dict:
        try:
            # Filtro 1: foto real
            real_probs = self._get_probs(image, self.real_photo_labels, self._real_photo_text)
            if real_probs[self.real_photo_labels[0]] < self.REAL_PHOTO_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "not_real_photo",
                    "detail": "La imagen parece ser un meme, captura de pantalla o imagen generada.",
                    "suggested_category": None,
                }

            # Filtro 2: exterior urbano
            outdoor_probs = self._get_probs(image, self.outdoor_labels, self._outdoor_text)
            if outdoor_probs[self.outdoor_labels[0]] < self.OUTDOOR_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "not_outdoor_urban",
                    "detail": "La imagen no muestra un espacio urbano exterior.",
                    "suggested_category": None,
                }

            # Filtro 3: tipo de problema + categoría sugerida
            problem_probs = self._get_probs(image, self.problem_labels, self._problem_text)
            problem_scores = {k: v for k, v in problem_probs.items() if k not in self.no_problem_labels}

            best_label = max(problem_scores, key=problem_scores.get)
            best_score = problem_scores[best_label]

            if best_score < self.PROBLEM_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "no_problem_detected",
                    "detail": "No se detectó un problema de infraestructura urbana claro.",
                    "suggested_category": None,
                }

            return {
                "valid": True,
                "rejection_reason": None,
                "detail": "Imagen válida con problema urbano detectable.",
                "suggested_category": self.label_to_category[best_label],
                "confidence": best_score,
                "scores": {
                    self.label_to_category.get(k, k): v
                    for k, v in problem_scores.items()
                },
            }

        except Exception as e:
            logger.error(f"Error clasificando imagen: {e}")
            return {
                "valid": False,
                "rejection_reason": "processing_error",
                "detail": "Error interno al procesar la imagen.",
                "suggested_category": None,
            }

    def compare_images(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calcula la similitud semántica entre dos imágenes usando CLIP.
        Devuelve un valor entre 0.0 (totalmente distintas) y 1.0 (idénticas).
        """
        if image1 is None or image2 is None:
            raise ValueError("Ambas imágenes son requeridas para comparar.")

        inputs = self.processor(images=[image1, image2], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalizar los vectores para comparar direcciones, no magnitudes
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Similitud coseno entre imagen 0 e imagen 1
        similarity = F.cosine_similarity(
            image_features[0].view(1, -1),
            image_features[1].view(1, -1)
        )

        return similarity.item()