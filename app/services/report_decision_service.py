# app/services/report_decision_service.py

class ReportDecisionService:
    def __init__(self, text_threshold=0.45):
        self.text_threshold = text_threshold 

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
            "reason": "La descripción no coincide con ningún problema municipal conocido."
        }