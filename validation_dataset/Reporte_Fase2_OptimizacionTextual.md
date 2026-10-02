# Reporte de Evaluación — Fase 2: Optimización Textual y Normalización Léxica
**Modelo:** Mendoza Reporta — motor de texto optimizado, sobre la misma lógica secuencial de la Fase 1

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), construidos específicamente para esta evaluación. Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías públicas por categoría de infraestructura urbana; los textos que acompañan a cada imagen son sintéticos, generados de forma programática para cubrir distintos registros (claro, indirecto, formal, vago) y niveles de ruido ortográfico/tipográfico (ninguno, leve, medio, fuerte). El detalle completo de la construcción se documenta en el Anexo A.

**Definición de métricas.** *Accuracy en reportes válidos* = válidos con la categoría final correcta / total de válidos (incluye tanto los rechazados por el filtro visual como las confusiones de categoría). *TNR (tasa de verdaderos negativos) en descartes* = descartes correctamente rechazados / total de descartes. Los intervalos de confianza son de Wilson al 95 %.

## 1. Resultados globales
- **Accuracy en reportes válidos:** 379/588 = 64.5% (IC95% 60.5–68.2%) (+5,3 puntos respecto de la Fase 1)
  - Errores por rechazo del filtro visual: 93 (15,8 %) — sin cambios, porque el filtro visual es idéntico al de la Fase 1
  - Errores por confusión de categoría: 116 (19,7 %), frente a 147 en la Fase 1
- **TNR en descartes:** 124/126 = 98.4% (IC95% 94.4–99.6%) — sin cambios

## 2. Diseño experimental
Se mantienen los umbrales de CLIP de la Fase 1 y la lógica "Texto OR Imagen". Los cambios se concentran exclusivamente en el componente de lenguaje natural:

1. **Normalización léxica** mediante expresiones regulares, aplicada antes de calcular la similitud de texto: corrige abreviaturas comunes (`q`→`que`, `x`→`por`, `d`→`de`, `xq`→`porque`, `tb`→`también`), errores ortográficos frecuentes (`zemaforo`→`semáforo`, `vache`→`bache`, `crter`→`cráter`, `ranpa`→`rampa`) y colapsa repeticiones de letras (`sucioooo`→`sucio`) **después** de aplicar las correcciones puntuales, para no destruir patrones que dependen de la repetición (por ejemplo, `rrttto`→`roto`).
2. **Refinamiento del mapa semántico de anclas:** se retiraron términos ambiguos y se agregaron anclas más específicas por categoría; en particular, se reformuló por completo el conjunto de anclas de *Agua y Cloacas* (`agua potable`, `caño maestro`, `pérdida de agua`, `alcantarilla hundida`, `desborde cloacal`, `olor podrido`) y se sumaron `puente roto`, `zanja` (Acequias y Drenajes) y `rompe cubiertas` (Baches y Pavimentación).

## 3. Discusión
- **La mejora se concentra casi por completo en una sola categoría.** El desglose por categoría muestra que *Agua y Cloacas* pasa de 19 a 52 aciertos (+33), mientras que el resto de las categorías suma en conjunto apenas −2 (Acequias −3, Alumbrado −2, Veredas −2, Baches −1, compensado por Plazas +4 y Semáforos +2). Esto indica que la ganancia de la Fase 2 proviene principalmente de la reformulación de las anclas semánticas, más que de la normalización léxica en sí.
- **El aporte específico de la normalización léxica no queda aislado en este experimento** (para eso sería necesario evaluar el mismo mapa semántico con y sin normalización); se retoma como línea de trabajo futuro.
- **El filtro visual sigue siendo el techo de la arquitectura:** como no cambia respecto de la Fase 1, los 93 rechazos por CLIP son un límite estructural que ninguna mejora textual puede superar mientras la lógica sea secuencial. Esto motiva la Fase 3.
