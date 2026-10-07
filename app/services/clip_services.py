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

        self.nsfw_labels = [
            "pornographic, explicit nudity, gore, violence, blood, weapons, or firearms",
            "safe, normal, everyday content",
        ]
        self.real_photo_labels = [
            "a real photograph taken with a camera outdoors",
            "a digital image, meme, screenshot, cartoon, drawing or AI generated image",
        ]
        self.outdoor_labels = [
            "an outdoor urban street scene, public infrastructure, road, a streetlight pole, or hanging cables against the sky",
            "an indoor scene inside a house, a close up of a person's face, a pet, or food",
        ]
        self.problem_labels = [
            "a blocked or flooded drainage ditch or canal on the street",       # Acequias y Drenajes
            "a broken or unlit streetlight, fallen electric pole, or broken hanging cables",             # Alumbrado Público
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
            "a broken or unlit streetlight, fallen electric pole, or broken hanging cables":             "Alumbrado Público",
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

        self.REAL_PHOTO_THRESHOLD = float(os.getenv("CLIP_REAL_PHOTO_THRESHOLD", 0.30))
        self.OUTDOOR_THRESHOLD    = float(os.getenv("CLIP_OUTDOOR_THRESHOLD", 0.25))
        self.PROBLEM_THRESHOLD    = float(os.getenv("CLIP_PROBLEM_THRESHOLD", 0.15))

        self._nsfw_features       = self._precompute_text_features(self.nsfw_labels)
        self._real_photo_features = self._precompute_text_features(self.real_photo_labels)
        self._outdoor_features    = self._precompute_text_features(self.outdoor_labels)
        self._problem_features    = self._precompute_text_features(self.problem_labels)

    def _precompute_text_features(self, labels: list[str]) -> torch.Tensor:
        """Tokeniza y precomputa características de texto normalizadas."""
        inputs = self.processor(text=labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            pooled_output = text_outputs.pooler_output
            text_features = self.model.text_projection(pooled_output)
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def classify_image(self, image: Image.Image) -> dict:
        try:
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

            with torch.no_grad():
                vision_outputs = self.model.vision_model(**image_inputs)
                pooled_output = vision_outputs.pooler_output
                image_features = self.model.visual_projection(pooled_output)
                
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()

            def get_probs(text_features, labels):
                logits = logit_scale * image_features @ text_features.T
                probs = logits.softmax(dim=1)[0]
                return dict(zip(labels, probs.tolist()))

            # Gate 1: NSFW
            nsfw_probs = get_probs(self._nsfw_features, self.nsfw_labels)
            if nsfw_probs[self.nsfw_labels[0]] > 0.85:
                return {
                    "valid": False,
                    "rejection_reason": "inappropriate_content",
                    "detail": "[V4] Contenido bloqueado: Alta probabilidad de violencia, armas o contenido explícito.",
                    "suggested_category": None,
                    "confidence": 0
                }

            # Gate 2: foto real
            real_probs = get_probs(self._real_photo_features, self.real_photo_labels)
            if real_probs[self.real_photo_labels[0]] < self.REAL_PHOTO_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "not_real_photo",
                    "detail": "[V4] La imagen parece ser un meme, captura de pantalla o imagen generada.",
                    "suggested_category": None,
                    "confidence": 0
                }

            # Gate 3: exterior urbano
            outdoor_probs = get_probs(self._outdoor_features, self.outdoor_labels)
            if outdoor_probs[self.outdoor_labels[0]] < self.OUTDOOR_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "not_outdoor_urban",
                    "detail": "[V4] La imagen no muestra un espacio urbano exterior.",
                    "suggested_category": None,
                    "confidence": 0
                }

            # Gate 4: tipo de problema
            problem_probs = get_probs(self._problem_features, self.problem_labels)
            
            no_prob_max = max(problem_probs.get(self.problem_labels[9], 0), problem_probs.get(self.problem_labels[10], 0))
            
            problem_scores = {k: v for k, v in problem_probs.items() if k not in self.no_problem_labels}
            best_label = max(problem_scores, key=problem_scores.get)
            best_score = problem_scores[best_label]
            
            # Competencia dinámica: si se parece más a una calle normal o algo irrelevante que a un problema
            if no_prob_max > best_score:
                return {
                    "valid": False,
                    "rejection_reason": "explicitly_no_problem",
                    "detail": "[V4] La imagen se detectó como un espacio sin problemas o una escena irrelevante.",
                    "suggested_category": None,
                    "confidence": 0
                }
                
            problem_scores = {k: v for k, v in problem_probs.items() if k not in self.no_problem_labels}
            best_label = max(problem_scores, key=problem_scores.get)
            best_score = problem_scores[best_label]

            if best_score < self.PROBLEM_THRESHOLD:
                return {
                    "valid": False,
                    "rejection_reason": "no_problem_detected",
                    "detail": "No se detectó un problema de infraestructura urbana claro.",
                    "suggested_category": None,
                    "confidence": 0
                }

            mapped_scores = {}
            for label, score in problem_scores.items():
                cat = self.label_to_category.get(label)
                if cat:
                    mapped_scores[cat] = score

            return {
                "valid": True,
                "rejection_reason": None,
                "detail": "Imagen válida con problema urbano detectable.",
                "suggested_category": self.label_to_category[best_label],
                "confidence": best_score,
                "scores": mapped_scores
            }

        except Exception as e:
            logger.error(f"Error clasificando imagen: {e}")
            return {
                "valid": False,
                "rejection_reason": "processing_error",
                "detail": "Error interno al procesar la imagen.",
                "suggested_category": None,
                "confidence": 0
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
            vision_outputs = self.model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output
            image_features = self.model.visual_projection(pooled_output)

        # Normalizar los vectores para comparar direcciones, no magnitudes
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Similitud coseno entre imagen 0 e imagen 1
        similarity = F.cosine_similarity(
            image_features[0].view(1, -1),
            image_features[1].view(1, -1)
        )

        return similarity.item()