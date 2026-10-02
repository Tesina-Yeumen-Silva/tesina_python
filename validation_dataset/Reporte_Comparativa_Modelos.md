# Reporte de Comparativa de Modelos Base (Pruebas y Validación)

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), construidos específicamente para esta evaluación. Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías públicas por categoría de infraestructura urbana; los textos que acompañan a cada imagen son sintéticos, generados de forma programática para cubrir distintos registros (claro, indirecto, formal, vago) y niveles de ruido ortográfico/tipográfico (ninguno, leve, medio, fuerte). El detalle completo de la construcción se documenta en el Anexo A.

## 1. Visión Artificial: OpenAI CLIP (ViT-B/32) vs. StreetCLIP (ViT-L/14)
Se evaluó el modelo visual adoptado por el sistema frente a un modelo especializado en escenas urbanas (StreetCLIP, preentrenado sobre imágenes de Google Street View para geolocalización), sobre las 714 imágenes del dataset. Se reportan dos familias de métricas: (a) con los umbrales fijos de producción (calibrados originalmente para OpenAI CLIP) y (b) métricas independientes del umbral (AUC, TPR a TNR ≥ 95 %, accuracy de categoría sin ningún filtro), que permiten una comparación equitativa entre modelos con distinta calibración.

| Métrica | OpenAI CLIP | StreetCLIP |
|---|---|---|
| TNR descartes (umbrales fijos) | 123/126 = 97.6% (IC95% 93.2–99.2%) | 122/126 = 96.8% (IC95% 92.1–98.8%) |
| TPR válidos (umbrales fijos) | 526/588 = 89.5% (IC95% 86.7–91.7%) | 532/588 = 90.5% (IC95% 87.8–92.6%) |
| Accuracy de categoría sin filtro (top-1) | 475/588 = 80.8% (IC95% 77.4–83.8%) | 501/588 = 85.2% (IC95% 82.1–87.8%) |
| Accuracy de categoría entre aceptados | 80,4 % | 84,6 % |
| AUC (score combinado / mejor probabilidad) | 0,994 / 0,990 | 0,994 / 0,987 |
| TPR @ TNR ≥ 95 % | 97,1 % | 95,9 % |
| Latencia media / p95 (ms, GPU Colab, 3 pasadas por imagen) | **80,6 / 210,5** | 404,4 / 422,2 |

**Lectura.** Con los mismos umbrales fijos, StreetCLIP no muestra menor sensibilidad: su TPR es incluso un punto mayor (90,5 % vs. 89,5 %) y el AUC es idéntico entre ambos modelos. StreetCLIP clasifica la categoría con mayor precisión (top-1: +4,4 puntos, 501 vs. 475 aciertos). La ventaja decisiva de OpenAI CLIP es la **latencia**, aproximadamente 5 veces menor, con una capacidad de filtrado equivalente. La elección de OpenAI CLIP para el sistema en producción se sostiene, por lo tanto, en un *trade-off* velocidad/precisión de categoría, y no en una mayor capacidad de rechazar contenido inválido.

## 2. Tolerancia Lingüística (NLP)
Se evaluó el clasificador de texto (similitud contra el mapa semántico de anclas, categoría de mayor similitud, sin aplicar umbral de aceptación) sobre los 588 textos válidos del dataset, agrupados según su estilo de redacción y su nivel de ruido tipográfico (cada texto pertenece a un único grupo). El baseline TF-IDF se ajustó únicamente con las anclas del mapa semántico, sin ver los textos de evaluación.

| Modelo | Claro, sin ruido (n=135) | Indirecto, sin anclas (n=138) | Formal (n=91) | Ruido medio/fuerte (n=141) | Vago (n=83) |
|---|---|---|---|---|---|
| Sentence-BERT (base BETO) + normalizador | 68,1 % (59,9–75,4) | **30,4 %** (23,4–38,6) | **49,5 %** (39,4–59,5) | 39,7 % (32,0–48,0) | 13,3 % |
| Sentence-BERT (base BETO) sin normalizador | 67,4 % (59,1–74,7) | 30,4 % (23,4–38,6) | 49,5 % (39,4–59,5) | 33,3 % (26,1–41,5) | 14,5 % |
| MiniLM multilingüe + normalizador | 51,9 % (43,5–60,1) | 23,2 % (16,9–30,9) | 30,8 % (22,2–40,9) | 22,7 % (16,6–30,3) | 8,4 % |
| MiniLM multilingüe sin normalizador | 51,1 % (42,8–59,4) | 22,5 % (16,3–30,1) | 27,5 % (19,4–37,4) | 20,6 % (14,7–28,0) | 9,6 % |
| TF-IDF por palabra | 77,8 % (70,1–84,0) | 5,8 % (3,0–11,0) | 40,7 % (31,1–50,9) | 24,1 % (17,8–31,8) | 8,4 % |
| TF-IDF por n-gramas de caracteres | **84,4 %** (77,4–89,6) | 17,4 % (12,0–24,6) | 42,9 % (33,2–53,1) | **43,3 %** (35,4–51,5) | 13,3 % |

**Lectura**
- **El grupo "Vago" funciona como control de validez del experimento:** todos los modelos quedan entre 8 % y 15 %, cerca del azar sobre 9 categorías (≈ 11 %), como corresponde a frases que no contienen información de categoría y que solo la imagen puede resolver.
- **Ningún método domina en todos los grupos.** Cuando el texto nombra explícitamente el problema (*Claro*), el TF-IDF por n-gramas de caracteres es el más preciso (84,4 %) y el TF-IDF por palabra supera al modelo semántico (77,8 % vs. 68,1 %): las anclas del mapa semántico son, en esencia, palabras clave, y el emparejamiento léxico las encuentra directamente. Cuando el texto describe el problema sin nombrarlo (*Indirecto*) o usa un registro formal, solo los modelos semánticos logran un desempeño apreciable, con BETO a la cabeza (30,4 % y 49,5 % respectivamente); el TF-IDF por palabra cae a 5,8 % en el grupo indirecto.
- **Con ruido tipográfico medio o fuerte**, el TF-IDF por n-gramas de caracteres (43,3 %) y BETO con normalizador (39,7 %) no se distinguen de forma estadísticamente significativa (los intervalos se solapan); ambos superan claramente a MiniLM y al TF-IDF por palabra.
- **El normalizador léxico aporta una mejora moderada y consistente, concentrada en el grupo con ruido:** +9 casos en BETO (33,3 % → 39,7 %) y +3 en MiniLM; en los demás grupos el efecto es de ±1 caso.
- **El modelo derivado de BETO supera a MiniLM en los cinco grupos.**
- **Ningún modelo supera el 50 % de precisión cuando el texto es indirecto o tiene ruido considerable**, lo que confirma que un clasificador de texto basado únicamente en anclas semánticas es, por sí solo, un componente débil frente a la variabilidad de reportes ciudadanos reales, y respalda la necesidad de la fusión con la imagen que se evalúa en la Fase 4.

## 3. Búsqueda Espacial y Rigor Geodésico: BallTree (Haversine) vs. KDTree vs. fuerza bruta
A la latitud de Mendoza (≈ 33° S), un grado de longitud mide aproximadamente 93,4 km frente a 111,3 km por grado de latitud (una contracción del 16,1 %). Se comparó, contra un *ground truth* de distancia Haversine calculado por fuerza bruta (200 consultas aleatorias, puntos sintéticos uniformemente distribuidos sobre un área de ≈ 10×10 km): BallTree con métrica Haversine, KDTree sobre coordenadas geográficas crudas (grados) y KDTree sobre una proyección local en metros.

| N | Radio | Vecinos reales | FN KDTree (grados) | FN BallTree | FN KDTree (metros) |
|---|---|---|---|---|---|
| 10 000 | 40 m | 79 | 13 (16,5 %) | 0 | 0 |
| 10 000 | 100 m | 597 | 92 (15,4 %) | 0 | 0 |
| 50 000 | 40 m | 454 | 62 (13,7 %) | 0 | 0 |
| 50 000 | 100 m | 3 041 | 481 (15,8 %) | 0 | 0 |
| 100 000 | 40 m | 997 | 152 (15,2 %) | 0 | 0 |
| 100 000 | 100 m | 6 005 | 971 (16,2 %) | 0 | 2 (0,03 %) |

**Lectura**
- **Un KDTree sobre coordenadas en grados, con un radio de búsqueda isótropo, pierde sistemáticamente entre el 14 % y el 16 % de los vecinos reales**, como consecuencia directa de la deformación elíptica descripta arriba.
- **BallTree con métrica Haversine y KDTree sobre coordenadas proyectadas en metros locales son, en la práctica, equivalentes en exactitud**; el KDTree proyectado fue, además, entre 1,5 y 2,5 veces más rápido en tiempo de consulta, aunque en ambos casos todas las consultas se resolvieron en menos de 0,03 ms.
- **La fuerza bruta vectorizada tardó hasta ≈ 5,4 ms por consulta con N = 100 000**: considerablemente más lenta que ambos árboles en términos relativos, pero no inviable para el volumen de datos actual del sistema.
- Los datos son sintéticos y uniformemente distribuidos, y las mediciones de tiempo corresponden al hardware disponible en el entorno de ejecución (Google Colab); ambos factores deben tenerse en cuenta al extrapolar los resultados a producción.
- La distancia Haversine asume una Tierra esférica; la diferencia frente a un modelo elipsoidal es inferior al 0,5 % a estas escalas, por lo que su uso como *ground truth* es adecuado para este dominio.

Ver Anexo A para el detalle de la construcción del dataset utilizado en las tres comparaciones de este reporte.
