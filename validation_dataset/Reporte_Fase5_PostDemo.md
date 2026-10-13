# Reporte de Evaluación — Fase 5: Calibración en Producción y Casos Borde (Post-Demo)
**Modelo:** Mendoza Reporta — Arquitectura de Calibración Final (Competencia Dinámica y Umbrales Flexibles)

**Dataset:** 714 elementos (588 reportes válidos y 126 descartes), idéntico al utilizado en las Fases 1 a 4 para garantizar comparabilidad estricta y trazabilidad experimental.
**Hardware de Ejecución:** Google Colab con acelerador GPU NVIDIA Tesla T4 (tiempo total de inferencia y ablación: ~1 min 45 s).

**Definición de métricas:**
- *Accuracy en reportes válidos* = reportes válidos con la categoría final correcta / total de válidos (penaliza tanto los falsos rechazos como las confusiones categoriales).
- *TNR (Tasa de Verdaderos Negativos) en descartes* = reportes de descarte correctamente rechazados / total de descartes.
- *Intervalos de confianza (IC95%):* Método de Wilson con corrección de continuidad al 95%.

---

## 1. Contexto y Hallazgos Cualitativos de la Demo en Vivo

Durante las demostraciones en vivo y pruebas de estrés en condiciones reales (*open-set*), la arquitectura de la Fase 4 (fusión multimodal lineal con umbrales fijos rígidos de CLIP, e.g., `PROBLEM_THRESHOLD = 0.32` y `OUTDOOR_THRESHOLD = 0.50`) evidenció limitaciones inherentes a los datasets de validación estáticos:

1. **Falsos Negativos por Rigidez (Problemas Sutiles):**
   Incidentes urbanos legítimos pero con baja huella visual (por ejemplo, cables caídos de baja tensión, luminarias de alumbrado público apagadas sobre fondo de cielo o baldosas sueltas) dispersan la probabilidad *Softmax* de CLIP entre múltiples clases candidatas. Sus puntajes fluctuaban entre `0.20` y `0.30`, quedando por debajo del corte estático `0.32`. Esto causaba el rechazo injusto de reportes ciudadanos legítimos (57 reportes válidos rechazados en la Fase 4).

2. **Vulnerabilidad a Falsos Positivos ante Relajación Ingenua (*Open-Set*):**
   Un intento directo de simplemente reducir los umbrales fijos abría la puerta a "vandalismo visual" y contenido fuera de dominio:
   - **Imágenes urbanas en buen estado ("The Beatles Test"):** Una fotografía de turistas cruzando una calle en perfecto estado (e.g. Abbey Road) alcanzaba puntajes residuales (~0.18 - 0.22) en varias categorías de infraestructura, siendo aceptada erróneamente si se bajaba el umbral sin un mecanismo de contraste.
   - **Contenido Ofensivo / Violento:** Fotografías con armas de fuego o elementos de violencia en la vía pública no eran filtradas de forma estricta por el clasificador NSFW genérico.

---

## 2. Ajuste Arquitectónico: Calibración Dinámica y Corrección de Umbrales

Para superar este dilema entre sensibilidad (capturar problemas sutiles) y especificidad (rechazar imágenes inválidas), se rediseñó el componente visual (`clip_services.py`) introduciendo tres cambios fundamentales:

### A. Corrección y Calibración de Umbrales Base
Se recalibraron los umbrales de corte de las compuertas visuales de CLIP. La evolución histórica de los umbrales a lo largo de las fases del proyecto es la siguiente:

| Umbral | Fases 1–2 (Línea Base) | Fases 3–4 (Soft Gating / Fusión) | Fase 5 (Post-Demo Calibrada) |
|---|:---:|:---:|:---:|
| `REAL_PHOTO_THRESHOLD` | 0,65 | 0,55 | **0,30** |
| `OUTDOOR_THRESHOLD` | 0,60 | 0,50 | **0,25** |
| `PROBLEM_THRESHOLD` | 0,40 | 0,32 | **0,15** |



### B. Competencia Dinámica (Dynamic Thresholding)
En lugar de depender exclusivamente de un umbral absoluto rígido para decidir si una imagen contiene un problema urbano, se implementó una **regla de competencia probabilística**:
Se contrastan los puntajes de las clases de infraestructura contra las etiquetas de normalidad urbana y contenido irrelevante:
$$\text{no\_prob\_max} = \max\left(P(\text{"calle o espacio público normal en buen estado"}), P(\text{"escena irrelevante sin problemas urbanos visible"})\right)$$

$$\text{best\_score} = \max_{c \in \text{Categorías}} P(c)$$

$$\textbf{Regla:} \quad \text{Si } \text{no\_prob\_max} > \text{best\_score} \implies \text{Rechazar con } \texttt{explicitly\_no\_problem}$$

Este mecanismo permite que un problema sutil con una confianza moderada (e.g., $0.22$) sea admitido siempre que la evidencia del problema supere a la de normalidad ($0.22 > 0.12$), pero rechaza de inmediato imágenes donde predomina la normalidad urbana ($0.20 < 0.65$ en una calle turística limpia).

### C. Compuerta de Seguridad Reforzada (Anti-Violencia y Armas)
Se amplió el vocabulario del Gate 1 Zero-Shot:
`"pornographic, explicit nudity, gore, violence, blood, weapons, or firearms"` (umbral $> 0.85$). Esto garantiza la interrupción inmediata del procesamiento ante imágenes que presenten riesgos a la seguridad pública antes de evaluar infraestructura.

---

## 3. Resultados Globales en el Benchmark Formal

Al ejecutar el pipeline completo de la Fase 5 sobre el dataset de prueba (714 muestras), se obtuvieron las siguientes métricas cuantitativas:

- **Accuracy en reportes válidos:** **456/588 = 77.6% (IC95% 74.0–80.7%)**  
  Representa un incremento de **+4,0 puntos porcentuales** respecto a la Fase 4 (73.6%) y de **+18,4 puntos** respecto a la línea base original (Fase 1: 59.2%).
- **Errores por rechazo del filtro visual:** **42 (7.1%)**  
  Se redujeron drásticamente desde los 57 errores (9.7%) de la Fase 4 y los 93 (15.8%) de las Fases 1 y 2. Esto representa el rescate efectivo de **15 reportes válidos adicionales** que antes eran bloqueados erróneamente.
- **Errores por confusión de categoría:** **90 (15.3%)**  
  Disminución respecto a los 98 errores (16.7%) de la Fase 4.
- **TNR en descartes (Especificidad):** **121/126 = 96.0% (IC95% 91.0–98.3%)**  
  Se mantiene un filtrado de descartes prácticamente intacto (solo 1 falso positivo adicional respecto a la Fase 4: 121 vs. 122).

---

## 4. Tabla Comparativa Consolidada: Fase 1 a Fase 5

### A. Resumen Global Acumulativo

| Métrica | Fase 1 (Línea Base) | Fase 2 (NLP) | Fase 3 (Soft-Gating) | Fase 4 (Fusión Fija) | Fase 5 (Post-Demo) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Accuracy en Válidos** | 59.2% (348/588) | 64.5% (379/588) | 68.2% (401/588) | 73.6% (433/588) | **77.6% (456/588)** |
| **TNR en Descartes** | 98.4% (124/126) | 98.4% (124/126) | 97.6% (123/126) | 96.8% (122/126) | **96.0% (121/126)** |
| **Errores por Rechazo** | 93 (15.8%) | 93 (15.8%) | 60 (10.2%) | 57 (9.7%) | **42 (7.1%)** |
| **Errores por Confusión** | 147 (25.0%) | 116 (19.7%) | 127 (21.6%) | 98 (16.7%) | **90 (15.3%)** |

### B. Rendimiento Desglosado por Categoría

| Categoría | Fase 1 | Fase 2 | Fase 3 | Fase 4 | Fase 5 | Δ Acumulativo (F1 → F5) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arbolado Público (n=68)** | 58,8 % | 58,8 % | 61,8 % | 69,1 % | **97.1% (66/68)** | **+38,3 %** |
| **Baches y Pavimentación (n=27)** | 63,0 % | 59,3 % | 59,3 % | 92,6 % | **96.3% (26/27)** | **+33,3 %** |
| **Plazas y Parques (n=96)** | 67,7 % | 71,9 % | 79,2 % | 83,3 % | **90.6% (87/96)** | **+22,9 %** |
| **Alumbrado Público (n=81)** | 56,8 % | 54,3 % | 59,3 % | 71,6 % | **85.2% (69/81)** | **+28,4 %** |
| **Agua y Cloacas (n=66)** | 28,8 % | 78,8 % | 80,3 % | 72,7 % | **74.2% (49/66)** | **+45,4 %** |
| **Limpieza y Residuos (n=63)** | 88,9 % | 88,9 % | 90,5 % | 95,2 % | **74.6% (47/63)** | −14,3 % |
| **Veredas y Accesibilidad (n=71)** | 52,1 % | 49,3 % | 53,5 % | 70,4 % | **67.6% (48/71)** | **+15,5 %** |
| **Semáforos y Señalización (n=89)** | 62,9 % | 65,2 % | 69,7 % | 64,0 % | **62.9% (56/89)** | 0,0 % |
| **Acequias y Drenajes (n=27)** | 44,4 % | 33,3 % | 33,3 % | 29,6 % | **29.6% (8/27)** | −14,8 % |

### C. Análisis de las Dinámicas Categoriales
- **Rescate Espectacular en Alumbrado y Arbolado:** La calibración de `OUTDOOR_THRESHOLD` (0.25) y `PROBLEM_THRESHOLD` (0.15) benefició de manera directa a *Alumbrado Público* (+13.6 puntos respecto a F4, alcanzando 85.2%) y a *Arbolado Público* (+28.0 puntos respecto a F4, alcanzando 97.1%). Fotografías de ramas altas, luminarias y cables colgando contra el cielo —que carecen de grandes superficies de asfalto o vereda— ya no son descartadas por falta de masa visual urbana.
- **El Trade-off en Limpieza y Residuos:** *Limpieza y Residuos* retrocedió de 95.2% a 74.6%. Al aumentar la sensibilidad visual hacia espacios verdes y árboles, acumulaciones de basura o ramas en veredas arboladas o parques son atraídas con mayor fuerza hacia *Plazas y Parques* o *Arbolado Público*, reflejando un solapamiento semántico real entre categorías urbanas.
- **Persistencia del Desafío en Acequias y Drenajes:** Con un 29.6%, permanece como la categoría con menor exactitud. Suelen confundirse con *Limpieza y Residuos* (cuando hay basura en la acequia) o con *Agua y Cloacas* (cuando hay agua estancada o desbordes).

---

## 5. Estudio de Ablación y Pruebas Estadísticas Pareadas (McNemar)

Para aislar el impacto del diseño arquitectónico en la Fase 5, se evaluaron cuatro variantes utilizando la caché de inferencias:

| Variante de Inferencia | Accuracy en Válidos | TNR en Descartes |
|---|:---:|:---:|
| **Solo Texto** (sin visión) | 165/588 = 28.1% (IC95% 24.6–31.8%) | 90/126 = 71.4% (IC95% 63.0–78.6%) |
| **Solo Imagen Fase 5** (CLIP con filtros dinámicos) | 443/588 = 75.3% (IC95% 71.7–78.7%) | 121/126 = 96.0% (IC95% 91.0–98.3%) |
| **Texto con Compuerta CLIP** (imagen solo filtra) | 427/588 = 72.6% (IC95% 68.9–76.1%) | 121/126 = 96.0% (IC95% 91.0–98.3%) |
| **Fusión Dinámica Fase 5** (Multimodal Ponderada) | **456/588 = 77.6% (IC95% 74.0–80.7%)** | **121/126 = 96.0% (IC95% 91.0–98.3%)** |

### Pruebas de McNemar (Significancia Estadística):
- **Fusión Fase 5 vs. Solo Imagen Fase 5:**  
  La Fusión supera a Solo Imagen en 15 casos válidos, mientras que Solo Imagen gana en 2 casos.  
  $$\chi^2 \text{ pareado} \implies p = 0.0023 \quad (\textbf{Significativo al nivel } p < 0.01)$$
  En descartes, el comportamiento es idéntico (0 vs. 0, $p = 1.0000$).
- **Fusión Fase 5 vs. Texto con Compuerta CLIP:**  
  La Fusión gana en 52 casos válidos frente a 23 de la compuerta.  
  $$\chi^2 \text{ pareado} \implies p = 0.0011 \quad (\textbf{Significativo al nivel } p < 0.01)$$

**Conclusión del Análisis Estadístico:** La ventaja de la Fusión Multimodal respecto a cualquier enfoque unimodal no es producto del azar muestral: la combinación de la señal textual normalizada con el peritaje de visión dinámico proporciona una ganancia estadísticamente significativa y reproducible.

---

## 6. Robustez y Desglose por Subgrupos

### A. Rendimiento según Estilo de Redacción del Ciudadano
| Estilo de Texto | Muestras ($n$) | Solo Texto | Solo Imagen | Texto c/ Gate | Fusión Fase 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Claro** | 192 | 53.1% | 77.6% | 80.2% | **83.9%** |
| **Indirecto** | 222 | 15.8% | 79.3% | 69.8% | **79.3%** |
| **Formal** | 91 | 29.7% | 67.0% | 67.0% | **68.1%** |
| **Vago** | 83 | 1.2% | 68.7% | 68.7% | **68.7%** |

*Observación:* Cuando el texto es completamente vago ("arreglen esto por favor", $n=83$), el modelo de texto por sí solo tiene una exactitud del 1.2% (inferior al azar puro de 11.1%), pero la arquitectura multimodal sostiene el acierto en un **68.7%** gracias al anclaje visual.

### B. Rendimiento según Nivel de Ruido Tipográfico
| Nivel de Ruido | Muestras ($n$) | Solo Texto | Solo Imagen | Texto c/ Gate | Fusión Fase 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ninguno** | 317 | 30.0% | 71.9% | 72.2% | **75.4%** |
| **Leve** | 103 | 26.2% | 82.5% | 75.7% | **83.5%** |
| **Medio** | 81 | 27.2% | 86.4% | 76.5% | **86.4%** |
| **Fuerte** | 87 | 24.1% | 69.0% | 66.7% | **70.1%** |

### C. Especificidad (TNR) por Grupo de Descarte
| Grupo de Contenido Inválido | Muestras ($n$) | Descarte Correcto (TNR) | Mecanismo Principal de Rechazo |
|---|:---:|:---:|---|
| **Interiores de Viviendas** | 33 | **100.0% (33/33)** | `not_outdoor_urban` |
| **Mascotas / Animales** | 33 | **100.0% (33/33)** | `not_real_photo` / `not_outdoor_urban` |
| **Montaña / Paisajes Naturales** | 29 | **96.6% (28/29)** | `not_outdoor_urban` |
| **Calles sin Problemas (Limpias)** | 31 | **87.1% (27/31)** | `explicitly_no_problem` (Dinámico) |

---

## 7. Verificación Cualitativa Post-Demo (Casos Borde en Vivo)

Las pruebas cualitativas con imágenes reales de casos borde confirmaron el éxito de los mecanismos introducidos:

1. **Resolución del "Beatles Test" (Abbey Road):**  
   Al enviar la icónica fotografía del cruce peatonal de Abbey Road (calle urbana limpia sin roturas), el clasificador asignó $P(\text{"normal street in good condition"}) = 0.58$, superando al mejor score de problema ($0.19$). El sistema rechazó la imagen bajo el código `explicitly_no_problem`, demostrando que los umbrales relajados no degradan el filtro ante escenas limpias.
2. **Bloqueo Inmediato de Violencia y Armas:**  
   Fotografías de prueba conteniendo armas u objetos ofensivos situados sobre la vereda activaron de inmediato la compuerta Gate 1 ($P > 0.85$), siendo rechazadas bajo `inappropriate_content` sin intentar categorizar infraestructura urbana.
3. **Rescate de Cables y Postes Colgando:**  
   Casos con luminarias rotas o cables caídos contra el cielo obtuvieron scores visuales de entre $0.22$ y $0.28$. Dado que la probabilidad de calle normal era baja ($0.08$), la competencia dinámica validó el incidente y la fusión clasificó correctamente como *Alumbrado Público*.

---

## 8. Conclusiones

La Fase 5 consolida el traspaso desde un prototipo experimental evaluado sobre un dataset estático hacia una **arquitectura de producción calibrada para escenarios abiertos (*open-set*)**.

1. **Evolución Histórica:** A lo largo de las cinco fases de investigación, el sistema incrementó su exactitud en reportes válidos del **59.2% al 77.6% (+18.4 puntos porcentuales)**, manteniendo al mismo tiempo un **TNR del 96.0%** en la detección de reportes inválidos y maliciosos.
2. **Superación del Techo del Filtro Visual:** Los falsos rechazos visuales se redujeron a más de la mitad (de 93 a 42 errores), demostrando que la competencia dinámica permite flexibilizar los umbrales sin sacrificar la seguridad del sistema.
3. **Validación Metodológica:** Los resultados experimentales obtenidos en el entorno de ejecución (GPU T4) respaldan plenamente las decisiones de diseño adoptadas en la demo en vivo, dejando el pipeline completamente calibrado y listo para su despliegue operativo en la plataforma municipal.
