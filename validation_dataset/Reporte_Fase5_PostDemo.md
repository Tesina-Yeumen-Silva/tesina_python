# Reporte de Evaluación — Fase 5: Calibración en Producción y Casos Borde (Post-Demo)
**Modelo:** Mendoza Reporta — Arquitectura Calibración Final (Competencia Dinámica)

**Contexto:** Este reporte documenta los hallazgos cualitativos y los ajustes arquitectónicos realizados tras someter el sistema a pruebas de estrés y demostraciones en vivo (escenarios *open-set*), los cuales evidenciaron las limitaciones del dataset de validación cerrado utilizado hasta la Fase 4.

---

## 1. El Problema Descubierto (Límites del Dataset Estático)

Aunque el enfoque de la Fase 4 (fusión multimodal con umbrales estáticos rígidos, ej. `PROBLEM_THRESHOLD = 0.35` y `OUTDOOR > 0.02`) arrojó métricas sólidas en el entorno controlado del dataset sintético (73.6% de Accuracy y 96.8% de TNR), el uso en vivo reveló dos grandes debilidades frente a escenarios reales:

- **Falsos Negativos por Rigidez (Problemas Sutiles):** Problemas urbanos válidos pero visualmente menos invasivos, como cables caídos (Alumbrado Público) o veredas levemente rotas, obtenían puntajes dispersos (~0.25 - 0.30). Esto se debe a que la distribución *Softmax* de CLIP fragmenta la probabilidad. Al usar un umbral rígido de 0.35, estos reportes eran rechazados injustamente.
- **Falsos Positivos por Vandalismo Visual (El desafío Open-Set):** Reducir rígidamente los umbrales para solucionar el punto anterior exponía al sistema a vulnerabilidades de "vandalismo visual" o imágenes irrelevantes que el dataset cerrado no contempló. Por ejemplo, al bajar el umbral, fotos callejeras turísticas (como "Los Beatles cruzando Abbey Road") o imágenes con armas de fuego en la acera lograban infiltrarse y ser catalogadas erróneamente como problemas urbanos.

---

## 2. Solución Arquitectónica: Modelo Calibración Final

Para lograr un equilibrio que permita recuperar los reportes sutiles sin abrir la puerta a imágenes irrelevantes o maliciosas, se ajustó el pipeline de validación visual (`clip_services.py`) implementando tres mecanismos clave:

1. **Relajación de Filtros Base (Contexto):** 
   Se disminuyeron los umbrales de imagen real y exterior urbano (de `0.02` a `0.005`) para evitar rechazos técnicos por anomalías menores de iluminación, filtros o calidad fotográfica de los usuarios.

2. **Competencia Dinámica (Dynamic Thresholding):** 
   Se abandonó el umbral rígido que definía si una calle era "normal" o no. En su lugar, se implementó una evaluación comparativa: se calcula el puntaje más alto entre las etiquetas de "descarte" (calle sin problemas o escena irrelevante) y se compara contra el "mejor problema detectado". 
   Si el score de la escena sin problemas supera al del problema (`no_prob_max > best_score`), la imagen se rechaza. Esto permite que el sistema acepte problemas con baja confianza absoluta (ej. 0.30), *siempre y cuando la evidencia del problema supere matemáticamente a la evidencia de normalidad*.

3. **Bloqueo Explícito Anti-Violencia:** 
   Se robusteció el filtro Zero-Shot de contenido inapropiado (NSFW), agregando explícitamente etiquetas de `"weapons, or firearms"`. Esto garantiza un aborto inmediato de la inferencia ante amenazas explícitas, bloqueándolas antes de que el modelo intente buscar infraestructura.

---

## 3. Resultados y Comportamiento Observado en Pruebas de Estrés

La implementación de la Fase 5 fue sometida a pruebas cualitativas con los casos borde que habían vulnerado a las fases anteriores, demostrando una altísima resiliencia:

- **Rescate de Falsos Negativos:** Reportes combinados (ej. Alumbrado 0.32 y Arbolado 0.35 en la misma imagen) que antes eran descartados por no llegar a 0.35, ahora son validados exitosamente gracias al umbral dinámico. Teóricamente, recuperar estos 57 errores por rechazo de la Fase 4 proyecta la exactitud (Accuracy) global hacia la banda del **78% - 80%**.
- **Rechazo Robusto de Irrelevancias ("The Beatles Test"):** Imágenes icónicas y turísticas de calles sin infraestructura dañada fueron rechazadas exitosamente bajo el código `explicitly_no_problem`, confirmando que la lógica dinámica mantiene un TNR altísimo sin recurrir a barreras fijas inalcanzables.
- **Seguridad Preventiva:** Fotografías conteniendo armas u objetos ofensivos en la vía pública fueron exitosa e inmediatamente rechazadas bajo `inappropriate_content`.

---

## 4. Conclusión

La transición de una lógica de "umbrales estáticos fijos" hacia un enfoque de **competencia probabilística dinámica** refleja la maduración del sistema para un entorno de producción real. 

Quedó evidenciado que los datasets cerrados, si bien son fundamentales para la calibración inicial, no pueden prever la infinita variabilidad de casos del mundo real (*open-set*). La arquitectura Calibración Final soluciona esta brecha, permitiendo que el sistema sea lo suficientemente sensible para captar deterioros urbanos sutiles y, simultáneamente, lo suficientemente robusto para rechazar contenido irrelevante o vandálico.
