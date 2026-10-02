# Reporte de Evaluación — Fase 1: Línea Base (Lógica Secuencial Rígida)
**Modelo:** Mendoza Reporta — arquitectura original de "compuertas duras" (*hard gating*)

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), construidos específicamente para esta evaluación. Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías públicas por categoría de infraestructura urbana; los textos que acompañan a cada imagen son sintéticos, generados de forma programática para cubrir distintos registros (claro, indirecto, formal, vago) y niveles de ruido ortográfico/tipográfico (ninguno, leve, medio, fuerte). El detalle completo de la construcción se documenta en el Anexo A.

**Definición de métricas.** *Accuracy en reportes válidos* = válidos con la categoría final correcta / total de válidos (incluye tanto los rechazados por el filtro visual como las confusiones de categoría). *TNR (tasa de verdaderos negativos) en descartes* = descartes correctamente rechazados / total de descartes. Los intervalos de confianza son de Wilson al 95 %.

## 1. Resultados globales
- **Accuracy en reportes válidos:** 348/588 = 59.2% (IC95% 55.2–63.1%)
  - Errores por rechazo del filtro visual: 93 (15,8 %)
  - Errores por confusión de categoría: 147 (25,0 %)
- **TNR en descartes:** 124/126 = 98.4% (IC95% 94.4–99.6%)

## 2. Resultados por categoría (válidos)

| Categoría | Aciertos | Accuracy |
|---|---|---|
| Limpieza y Residuos | 56/63 | 88,9 % |
| Plazas y Parques | 65/96 | 67,7 % |
| Baches y Pavimentación | 17/27 | 63,0 % |
| Semáforos y Señalización | 56/89 | 62,9 % |
| Arbolado Público | 40/68 | 58,8 % |
| Alumbrado Público | 46/81 | 56,8 % |
| Veredas y Accesibilidad | 37/71 | 52,1 % |
| Acequias y Drenajes | 12/27 | 44,4 % |
| Agua y Cloacas | 19/66 | 28,8 % |

## 3. Diseño experimental
Esta fase evalúa la arquitectura secuencial de triaje sobre la que se construyen las siguientes iteraciones. La lógica es estricta: la imagen se evalúa primero con OpenAI CLIP (ViT-B/32); si el filtro visual la rechaza, el reporte se descarta sin considerar el texto. Si la imagen es aceptada, la categoría final se decide por texto **o** por imagen, nunca por ambos a la vez ("Texto OR Imagen").

- **Filtro visual:** `REAL_PHOTO_THRESHOLD = 0.65`, `OUTDOOR_THRESHOLD = 0.60`, `PROBLEM_THRESHOLD = 0.40`.
- **Motor de texto:** sentence-transformer en español (`hiiamsid/sentence_similarity_spanish_es`) contra un mapa semántico de anclas por categoría, sin normalización léxica.
- **Regla de decisión:** si la similitud de texto supera 0,45, decide el texto; en caso contrario, decide la categoría sugerida por CLIP.

## 4. Discusión
- **El filtro de descartes es muy efectivo (98,4 %)** frente al conjunto de ataques evaluado (imágenes irrelevantes —interiores, paisajes, mascotas y calles sin problemas visibles— acompañadas de texto engañoso, vago o sin relación).
- **La mayoría de los errores en válidos son confusiones de categoría (25,0 %), no rechazos del filtro visual (15,8 %).** El texto, en su configuración base, no logra corregir estas confusiones porque compite con la imagen en lugar de complementarla.
- **Agua y Cloacas (28,8 %) y Acequias y Drenajes (44,4 %) son las categorías más débiles.** Sus imágenes (calles/veredas inundadas y acumulación de residuos, respectivamente) son semánticamente cercanas a otras categorías del sistema, lo que produce confusiones sistemáticas que se analizan con más detalle en la Fase 4.
- Estos resultados constituyen la línea base sobre la que se miden las mejoras de las Fases 2 a 4.
