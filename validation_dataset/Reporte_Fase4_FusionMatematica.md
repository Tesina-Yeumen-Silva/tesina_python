# Reporte de Evaluación — Fase 4: Fusión Multimodal Lineal
**Modelo:** Mendoza Reporta — arquitectura final

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), construidos específicamente para esta evaluación. Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías públicas por categoría de infraestructura urbana; los textos que acompañan a cada imagen son sintéticos, generados de forma programática para cubrir distintos registros (claro, indirecto, formal, vago) y niveles de ruido ortográfico/tipográfico (ninguno, leve, medio, fuerte). El detalle completo de la construcción se documenta en el Anexo A.

**Definición de métricas.** *Accuracy en reportes válidos* = válidos con la categoría final correcta / total de válidos (incluye tanto los rechazados por el filtro visual como las confusiones de categoría). *TNR (tasa de verdaderos negativos) en descartes* = descartes correctamente rechazados / total de descartes. Los intervalos de confianza son de Wilson al 95 %.

## 1. Resultados globales
- **Accuracy en reportes válidos:** 433/588 = 73.6% (IC95% 69.9–77.0%) (+5,4 puntos respecto de la Fase 3)
  - Errores por rechazo: 57 (9,7 %)
  - Errores por confusión de categoría: 98 (16,7 %)
- **TNR en descartes:** 122/126 = 96.8% (IC95% 92.1–98.8%)

## 2. Resultados por categoría, a lo largo de las cuatro fases

| Categoría | Fase 1 | Fase 2 | Fase 3 | Fase 4 |
|---|---|---|---|---|
| Limpieza y Residuos (n=63) | 88,9 % | 88,9 % | 90,5 % | **95,2 %** |
| Baches y Pavimentación (n=27) | 63,0 % | 59,3 % | 59,3 % | **92,6 %** |
| Plazas y Parques (n=96) | 67,7 % | 71,9 % | 79,2 % | **83,3 %** |
| Agua y Cloacas (n=66) | 28,8 % | 78,8 % | 80,3 % | 72,7 % |
| Alumbrado Público (n=81) | 56,8 % | 54,3 % | 59,3 % | **71,6 %** |
| Veredas y Accesibilidad (n=71) | 52,1 % | 49,3 % | 53,5 % | **70,4 %** |
| Arbolado Público (n=68) | 58,8 % | 58,8 % | 61,8 % | **69,1 %** |
| Semáforos y Señalización (n=89) | 62,9 % | 65,2 % | 69,7 % | 64,0 % |
| Acequias y Drenajes (n=27) | 44,4 % | 33,3 % | 33,3 % | 29,6 % |

## 3. Diseño experimental
Sobre los umbrales relajados de la Fase 3, se incorporan dos mecanismos:

1. **Rescate por texto:** si el filtro visual rechaza un reporte únicamente por `no_problem_detected` (imagen ambigua, no claramente dañina ni claramente normal) y el texto es válido, el reporte no se descarta y pasa a la fusión.
2. **Fusión lineal ponderada:** `score = 0,6 · similitud_texto + 0,4 · probabilidad_CLIP`, combinando ambas señales por categoría; se elige la de mayor score combinado. Por diseño, la fusión nunca rechaza un reporte — el rechazo lo determinan exclusivamente los filtros de seguridad previos.
3. **Filtros anti-vandalismo explícitos:** bloqueo directo si CLIP asigna más de 0,40 de probabilidad a "escena sin ningún problema visible" o a "contenido inapropiado", independientemente de lo que diga el texto.

## 4. Estudio de ablación
Se compararon cuatro variantes sobre el mismo conjunto de predicciones cacheadas de CLIP y del motor de texto, para aislar el aporte de cada componente:

| Variante | Accuracy en válidos | TNR en descartes |
|---|---|---|
| Solo texto (sin filtro visual) | 148/588 = 25.2% (IC95% 21.8–28.8%) | 90/126 = 71.4% (IC95% 63.0–78.6%) |
| Solo imagen (CLIP con sus filtros) | 423/588 = 71.9% (IC95% 68.2–75.4%) | 123/126 = 97.6% (IC95% 93.2–99.2%) |
| Texto con compuerta visual (CLIP solo filtra; la categoría la decide el texto) | 401/588 = 68.2% (IC95% 64.3–71.8%) | 122/126 = 96.8% (IC95% 92.1–98.8%) |
| **Fusión lineal (0,6 / 0,4)** | 433/588 = 73.6% (IC95% 69.9–77.0%) | 122/126 = 96.8% (IC95% 92.1–98.8%) |

**Pruebas de McNemar (pareadas, sobre reportes válidos):**
- Fusión vs. solo imagen: la fusión gana en 14 casos, solo imagen gana en 4, **p = 0,031**. En descartes, 0 vs. 1 (p = 1,0).
- Fusión vs. texto con compuerta visual: la fusión gana en 53 casos, la otra variante en 21, **p = 0,0003**.

La comparación fusión vs. solo imagen es significativa al 5 %, aunque su robustez se reduce si se corrige por comparaciones múltiples (Bonferroni con 2 pruebas: p ≈ 0,06). Se trata de evidencia moderada de una mejora pequeña: ambas variantes coinciden en 570 de los 588 reportes válidos.

## 5. Calibración del peso de fusión
Se dividió el conjunto en dos mitades estratificadas (calibración y prueba) para evaluar si el peso 0,6/0,4 es una elección informada o arbitraria.

- En la mitad de calibración, la exactitud balanceada es prácticamente plana para `w_text` entre 0,0 y 0,8 (0,863–0,868) y solo cae en 0,9–1,0 (0,846 y 0,839). El valor que maximiza la métrica (0,1) no es distinguible de 0,6.
- En la mitad de prueba: `w_text=0,1` da 207/294 = 70.4% (IC95% 65.0–75.3%) en válidos y 60/63 = 95.2% (IC95% 86.9–98.4%) en descartes; `w_text=0,6` da 212/294 = 72.1% (IC95% 66.7–76.9%) en válidos y 60/63 = 95.2% (IC95% 86.9–98.4%) en descartes. El peso seleccionado en calibración no mejora en la mitad de prueba.

**Conclusión:** el resultado final es prácticamente insensible al peso de fusión dentro de un rango amplio; el desempeño está dominado por el filtro visual, y 0,6/0,4 es una elección razonable pero no demostrablemente óptima.

## 6. Desglose por tipo de texto e imagen

**Reportes válidos, por estilo de texto:**

| Estilo | n | Solo texto | Solo imagen | Texto+compuerta | Fusión |
|---|---|---|---|---|---|
| Claro | 192 | 50,5 % | 74,0 % | 72,9 % | **77,6 %** |
| Formal | 91 | 27,5 % | 67,0 % | 70,3 % | **70,3 %** |
| Indirecto | 222 | 11,3 % | 73,9 % | 63,5 % | 73,9 % |
| Vago | 83 | 1,2 % | 67,5 % | 67,5 % | 67,5 % |

**Reportes válidos, por nivel de ruido textual:**

| Ruido | n | Solo texto | Solo imagen | Texto+compuerta | Fusión |
|---|---|---|---|---|---|
| Ninguno | 317 | 27,1 % | 69,4 % | 69,1 % | **72,9 %** |
| Leve | 103 | 24,3 % | 78,6 % | 71,8 % | 77,7 % |
| Medio | 81 | 22,2 % | 80,2 % | 65,4 % | 80,2 % |
| Fuerte | 87 | 21,8 % | 65,5 % | 63,2 % | 65,5 % |

**Reportes válidos, por contenido de la imagen:**

| Contenido | n | Solo texto | Solo imagen | Texto+compuerta | Fusión |
|---|---|---|---|---|---|
| Acequias (imágenes de residuos acumulados) | 27 | 18,5 % | 29,6 % | 33,3 % | 29,6 % |
| Alumbrado | 81 | 16,0 % | 72,8 % | 59,3 % | 71,6 % |
| Arbolado | 68 | 30,9 % | 69,1 % | 61,8 % | 69,1 % |
| Baches | 27 | 22,2 % | 92,6 % | 59,3 % | 92,6 % |
| Cloacas (calles/veredas inundadas) | 66 | 68,2 % | 60,6 % | 78,8 % | 72,7 % |
| Limpieza | 63 | 27,0 % | 93,7 % | 90,5 % | 95,2 % |
| Plazas: bancos | 34 | 5,9 % | 85,3 % | 79,4 % | 85,3 % |
| Plazas: juegos | 62 | 1,6 % | 82,3 % | 79,0 % | 82,3 % |
| Semáforos: carteles de tránsito | 32 | 6,2 % | 78,1 % | 65,6 % | 71,9 % |
| Semáforos: semáforos | 57 | 43,9 % | 54,4 % | 71,9 % | 59,6 % |
| Veredas: rampas de accesibilidad | 11 | 36,4 % | 54,5 % | 54,5 % | 54,5 % |
| Veredas: roturas | 60 | 11,7 % | 71,7 % | 55,0 % | 73,3 % |

**Descartes (TNR), por tipo de imagen y de texto:**

| Grupo | n | Solo texto | Solo imagen | Texto+compuerta | Fusión |
|---|---|---|---|---|---|
| Calle sin problema visible | 31 | 67,7 % | 90,3 % | 87,1 % | 87,1 % |
| Interiores | 33 | 75,8 % | 100 % | 100 % | 100 % |
| Paisaje de montaña | 29 | 65,5 % | 100 % | 100 % | 100 % |
| Mascotas | 33 | 75,8 % | 100 % | 100 % | 100 % |
| Texto con trampa (claro) | 35 | 54,3 % | 100 % | 97,1 % | 97,1 % |
| Texto con trampa (indirecto) | 48 | 66,7 % | 93,8 % | 93,8 % | 93,8 % |
| Texto vago | 20 | 85,0 % | 100 % | 100 % | 100 % |
| Texto irrelevante | 23 | 95,7 % | 100 % | 100 % | 100 % |

## 7. Discusión
- **La fusión mejora sobre solo imagen de forma pequeña pero probablemente real** (+10 reportes netos, p = 0,031), y mejora de forma clara sobre dejar que el texto decida solo (p = 0,0003).
- **La ganancia de la fusión frente a solo imagen proviene exclusivamente de los reportes con texto claro, formal o sin ruido.** Con texto claro la fusión suma +3,6 puntos sobre solo imagen (77,6 % vs. 74,0 %) y con texto formal +3,3 puntos; con texto indirecto o vago, la fusión **iguala** a solo imagen (no la perjudica, pero tampoco la mejora). Por nivel de ruido, la ganancia neta de +10 reportes se concentra enteramente en el grupo sin ruido (+11); con ruido leve, medio o fuerte la fusión no aporta sobre solo imagen.
- **Dejar que el texto decida en solitario (con el filtro visual solo como compuerta) perjudica el desempeño** frente a solo imagen (68,2 % vs. 71,9 %): en textos indirectos cae de 73,9 % a 63,5 %, y en Baches de 92,6 % a 59,3 %. La fusión evita sistemáticamente este daño.
- **En dos categorías donde el filtro visual es estructuralmente débil, el texto aporta una mejora sustancial que la fusión solo recupera parcialmente.** En Cloacas, el texto solo (68,2 %) ya supera a la imagen sola (60,6 %), y el texto con compuerta llega a 78,8 %; la fusión con peso fijo 0,6/0,4 se queda en 72,7 %. En Semáforos, el texto con compuerta alcanza 71,9 % frente al 54,4 % de la imagen sola; la fusión llega solo a 59,6 %. Esto sugiere que una **fusión con peso adaptativo** (mayor peso al texto cuando la confianza de CLIP es baja, o pesos calibrados por categoría) podría capturar una ganancia adicional que la ponderación fija no aprovecha.
- **El costo en descartes es acotado y concentrado en un único vector de ataque.** Los 4 descartes que la fusión no detecta corresponden todos a fotografías de calles sin problemas visibles acompañadas de un texto plausible (TNR 87,1 % sobre 31 casos de este tipo); interiores, paisajes de montaña y mascotas se rechazan siempre, independientemente del texto. Solo imagen deja pasar 3 de esos mismos 31 casos (diferencia no significativa, p = 1,0).
- **Acequias y Drenajes (29,6 %) es la única categoría que no mejora en ninguna fase.** Sus imágenes corresponden a acumulación de residuos, visualmente indistinguibles para CLIP de la categoría Limpieza y Residuos; es una limitación de la definición de categorías del dominio, no del algoritmo de fusión. Veredas: rampas cuenta con solo 11 observaciones, insuficientes para conclusiones robustas.
- **El umbral de aceptación del texto podría estar recortando aciertos disponibles.** Evaluado sin ningún umbral, el motor de texto (BETO con normalización) acierta el 41,8 % de los 588 casos válidos; la variante "solo texto" de esta ablación, que sí aplica el umbral de 0,45, llega a 25,2 %. Un barrido sistemático de ese umbral queda como trabajo futuro.

## 8. Limitaciones y trabajo futuro
1. Las imágenes provienen de búsquedas web y son, en general, nítidas y representativas de cada categoría; fotografías ciudadanas reales (tomadas de noche, en movimiento o con mala iluminación) plantearían un desafío mayor al filtro visual, y el rol relativo del texto en la fusión podría aumentar. Validar con un conjunto de fotografías ciudadanas reales es la extensión más importante de este trabajo.
2. Los textos son sintéticos; su generación se diseñó para cubrir un rango amplio de estilos y ruido (Anexo A), pero deben contrastarse con reportes ciudadanos reales en una segunda etapa.
3. Los descartes cubren cuatro tipos de imagen y cuatro estilos de texto engañoso; no incluyen contenido ofensivo ni capturas de pantalla, dos vectores de descarte relevantes en un sistema en producción.
4. Quedan como trabajo futuro: una fusión con peso adaptativo (por confianza de CLIP o por categoría), un barrido del umbral de aceptación del texto, una ablación aislada del normalizador léxico, y una revisión de la definición de las categorías Acequias y Drenajes y Agua y Cloacas.
