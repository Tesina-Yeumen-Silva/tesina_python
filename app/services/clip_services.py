from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

class ClipService:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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

        self.REAL_PHOTO_THRESHOLD = 0.65
        self.OUTDOOR_THRESHOLD = 0.60
        self.PROBLEM_THRESHOLD = 0.40

    def _get_probs(self, image: Image.Image, labels: list[str]) -> dict:
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        return dict(zip(labels, probs.tolist()))

    def classify_image(self, image: Image.Image) -> dict:
        # Filtro 1: foto real
        real_probs = self._get_probs(image, self.real_photo_labels)
        if real_probs[self.real_photo_labels[0]] < self.REAL_PHOTO_THRESHOLD:
            return {
                "valid": False,
                "rejection_reason": "not_real_photo",
                "detail": "La imagen parece ser un meme, captura de pantalla o imagen generada.",
                "suggested_category": None,
            }

        # Filtro 2: exterior urbano
        outdoor_probs = self._get_probs(image, self.outdoor_labels)
        if outdoor_probs[self.outdoor_labels[0]] < self.OUTDOOR_THRESHOLD:
            return {
                "valid": False,
                "rejection_reason": "not_outdoor_urban",
                "detail": "La imagen no muestra un espacio urbano exterior.",
                "suggested_category": None,
            }

        # Filtro 3: tipo de problema + categoría sugerida
        problem_probs = self._get_probs(image, self.problem_labels)

        no_problem_labels = {
            "a normal street or public space in good condition with no issues",
            "an unrelated scene with no urban infrastructure problems visible",
        }

        problem_scores = {k: v for k, v in problem_probs.items() if k not in no_problem_labels}
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
    
    def compare_images(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calcula la similitud semántica entre dos imágenes usando CLIP.
        Devuelve un valor entre 0.0 (totalmente distintas) y 1.0 (idénticas).
        """
        # Procesamos ambas imágenes a la vez
        inputs = self.processor(images=[image1, image2], return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalizamos los vectores para comparar
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Calculamos la similitud del coseno entre la imagen 0 y la imagen 1
        similarity = F.cosine_similarity(image_features[0].view(1, -1), image_features[1].view(1, -1))
        
        return similarity.item()