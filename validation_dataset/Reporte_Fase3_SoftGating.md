# Reporte de Evaluación — Fase 3: Soft Gating (Relajación de Umbrales Visuales)
**Modelo:** Mendoza Reporta — umbrales visuales relajados, sobre la misma lógica secuencial

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), construidos específicamente para esta evaluación. Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías públicas por categoría de infraestructura urbana; los textos que acompañan a cada imagen son sintéticos, generados de forma programática para cubrir distintos registros (claro, indirecto, formal, vago) y niveles de ruido ortográfico/tipográfico (ninguno, leve, medio, fuerte). El detalle completo de la construcción se documenta en el Anexo A.

**Definición de métricas.** *Accuracy en reportes válidos* = válidos con la categoría final correcta / total de válidos (incluye tanto los rechazados por el filtro visual como las confusiones de categoría). *TNR (tasa de verdaderos negativos) en descartes* = descartes correctamente rechazados / total de descartes. Los intervalos de confianza son de Wilson al 95 %.

## 1. Resultados globales
- **Accuracy en reportes válidos:** 401/588 = 68.2% (IC95% 64.3–71.8%) (+3,7 puntos respecto de la Fase 2)
  - Errores por rechazo del filtro visual: 60 (10,2 %), frente a 93 en la Fase 2
  - Errores por confusión de categoría: 127 (21,6 %), frente a 116 en la Fase 2
- **TNR en descartes:** 123/126 = 97.6% (IC95% 93.2–99.2%) (−1 descarte respecto de la Fase 2)

## 2. Diseño experimental
Se relajan los tres umbrales del filtro visual de CLIP, manteniendo sin cambios el motor de texto de la Fase 2 y la lógica secuencial "Texto OR Imagen" (**sin fusión**: el texto solo puede decidir sobre imágenes que el filtro visual ya dejó pasar).

| Umbral | Fases 1–2 | Fase 3 |
|---|---|---|
| `REAL_PHOTO_THRESHOLD` | 0,65 | 0,55 |
| `OUTDOOR_THRESHOLD` | 0,60 | 0,50 |
| `PROBLEM_THRESHOLD` | 0,40 | 0,32 |

## 3. Discusión
- **Se recuperan 33 reportes que antes rechazaba el filtro visual**, pero 11 de ellos terminan en una confusión de categoría en lugar de un acierto, para una ganancia neta de 22 reportes (+3,7 puntos). Las mayores ganancias por categoría son Plazas y Parques (+7), Alumbrado Público (+4), Semáforos y Señalización (+4) y Veredas y Accesibilidad (+3).
- **El costo en descartes es mínimo:** un descarte adicional pasa el filtro (123 de 126, frente a 124 de 126 en las Fases 1–2).
- **El cuello de botella se desplaza del filtro visual a la decisión de categoría.** Relajar los umbrales convierte rechazos en oportunidades para el motor de texto, pero como la lógica sigue siendo "Texto OR Imagen" (competencia, no colaboración), parte de esas oportunidades se pierden como confusiones de categoría. Este resultado motiva directamente la fusión multimodal de la Fase 4.
