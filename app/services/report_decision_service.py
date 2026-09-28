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
            return {"category": best_category, "confidence": best_score, "valid": True}
        return {"category": None, "confidence": best_score, "valid": False}

    def fuse_decisions(self, text_scores: dict, clip_result: dict) -> dict:
        text_decision = self.get_best_text_category(text_scores)
        clip_category = clip_result.get("suggested_category")
        clip_scores = clip_result.get("scores", {})

        if not text_decision["valid"]:
            return {
                "category": clip_category, "confidence": clip_result.get("confidence", 0.0),
                "winner": "IMAGEN", "motivo": "Predominancia visual", "valid": clip_result.get("valid", False)
            }
        
        if not clip_scores:
            return {
                "category": text_decision["category"], "confidence": text_decision["confidence"],
                "winner": "TEXTO", "motivo": "Texto validado sin imagen", "valid": True
            }

        all_cats = set(text_scores.keys()).union(set(clip_scores.keys()))
        fused = {}
        for cat in all_cats:
            fused[cat] = (self.text_weight * text_scores.get(cat, 0.0)) + (self.clip_weight * clip_scores.get(cat, 0.0))

        best_cat = max(fused, key=fused.get)
        best_fused_score = fused[best_cat]
        is_valid = True

        if text_decision["category"] == clip_category:
            winner, motivo = "CONSENSO", "consenso texto-imagen"
        elif best_cat == text_decision["category"]:
            winner, motivo = "TEXTO_PREDOMINANTE", "fusión guiada por texto"
        else:
            winner, motivo = "IMAGEN_PREDOMINANTE", "fusión guiada por imagen"

        return {"category": best_cat, "confidence": best_fused_score, "winner": winner, "motivo": motivo, "valid": is_valid}