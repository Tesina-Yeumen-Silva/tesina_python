# app/services/report_decision_service.py

class ReportDecisionService:
    def __init__(self, text_threshold=0.45, text_weight=0.6, clip_weight=0.4):
        self.text_threshold = text_threshold 
        self.text_weight = text_weight
        self.clip_weight = clip_weight

    def get_best_text_category(self, scores: dict) -> dict:
        if not scores:
            return {"category": None, "confidence": 0, "valid": False}
            
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score >= self.text_threshold:
            return {
                "category": best_category,
                "confidence": best_score,
                "valid": True
            }
        
        return {
            "category": None,
            "confidence": best_score,
            "valid": False,
            "reason": "La descripción no coincide con ningún problema conocido."
        }

    def fuse_decisions(self, text_scores: dict, clip_result: dict) -> dict:
        """
        Fusión ponderada multimodal (Texto + Imagen).
        Combina las probabilidades de ambas modalidades para robustecer la clasificación
        frente a ambigüedades o errores de ortografía.
        """
        text_decision = self.get_best_text_category(text_scores)
        clip_category = clip_result.get("suggested_category")
        clip_scores = clip_result.get("scores", {})

        # Si el texto no es válido o está vacío, predomina la imagen (CLIP)
        if not text_decision["valid"]:
            return {
                "category": clip_category,
                "confidence": clip_result.get("confidence", 0.0),
                "winner": "IMAGEN",
                "motivo": "análisis visual de la imagen (texto insuficiente o ambiguo)",
                "scores": clip_scores
            }

        # Si no hay imagen válida o no tiene scores, predomina el texto
        if not clip_result.get("valid") or not clip_scores:
            return {
                "category": text_decision["category"],
                "confidence": text_decision["confidence"],
                "winner": "TEXTO",
                "motivo": "descripción del incidente (sin imagen urbana válida)",
                "scores": text_scores
            }

        # Calibración de scores: Convertir similitud coseno de texto a distribución de probabilidad
        # utilizando Softmax con temperatura para nivelar con la escala de CLIP (Softmax)
        import math
        temperature = 0.12
        exp_text = {cat: math.exp(max(0.0, score) / temperature) for cat, score in text_scores.items()}
        sum_exp = sum(exp_text.values())
        norm_text_scores = {cat: (val / sum_exp) if sum_exp > 0 else 0.0 for cat, val in exp_text.items()}

        # Fusión ponderada de probabilidades calibradas
        all_categories = set(text_scores.keys()).union(set(clip_scores.keys()))
        fused_scores = {}
        for cat in all_categories:
            t_prob = norm_text_scores.get(cat, 0.0)
            c_prob = clip_scores.get(cat, 0.0)
            fused_scores[cat] = (self.text_weight * t_prob) + (self.clip_weight * c_prob)

        best_category = max(fused_scores, key=fused_scores.get)
        best_fused_score = fused_scores[best_category]

        # Determinar motivo explicativo
        t_winner = text_decision["category"]
        c_winner = clip_category

        if t_winner == c_winner:
            motivo = f"consenso pleno entre texto e imagen ({best_category}, score: {best_fused_score:.2f})"
            winner = "CONSENSO"
        elif best_category == t_winner:
            motivo = f"fusión multimodal con predominancia textual ({best_category}, texto: {text_scores.get(best_category, 0):.2f}, clip: {clip_scores.get(best_category, 0):.2f})"
            winner = "TEXTO_PREDOMINANTE"
        else:
            motivo = f"fusión multimodal corregida por imagen ({best_category}, clip: {clip_scores.get(best_category, 0):.2f}, texto: {text_scores.get(best_category, 0):.2f})"
            winner = "IMAGEN_PREDOMINANTE"

        return {
            "category": best_category,
            "confidence": best_fused_score,
            "winner": winner,
            "motivo": motivo,
            "scores": fused_scores
        }