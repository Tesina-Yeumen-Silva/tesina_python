# Anexo A — Construcción del Dataset de Evaluación

Este anexo documenta la metodología utilizada para construir el dataset de 714 elementos (588 reportes válidos y 126 descartes) sobre el que se ejecutan los cinco notebooks de evaluación (Fases 1 a 4 y la Comparativa de Modelos Base).

## A.1. Selección y organización de imágenes

Las imágenes se obtuvieron mediante búsqueda automatizada de fotografías de dominio público (DuckDuckGo Image Search), con una consulta de búsqueda distinta por categoría de infraestructura urbana. Cada carpeta de categoría se revisó manualmente para eliminar imágenes que no correspondían al concepto buscado, por lo que la numeración de archivos dentro de cada carpeta no es necesariamente correlativa.

| Carpeta | Contenido |
|---|---|
| `acequias` | Acumulación de residuos y basura |
| `alumbrado` | Postes de luz rotos o caídos |
| `arbolado` | Árboles o troncos caídos |
| `baches` | Agujeros y roturas en la calzada |
| `cloacas` | Calles o veredas inundadas |
| `limpieza` | Basura o contenedores desbordados en la vía pública |
| `plazas` (0001–0034) | Bancos de plaza rotos |
| `plazas` (0035–0101) | Juegos infantiles de plaza rotos |
| `semaforos` (0001–0032) | Semáforos rotos o fuera de servicio |
| `semaforos` (0033–0067) | Carteles de señalización vial caídos o rotos |
| `semaforos` (0068–0099) | Semáforos rotos o fuera de servicio |
| `veredas` (0001–0033) | Veredas con grietas o agujeros |
| `veredas` (0034–0063) | Rampas de accesibilidad rotas |
| `veredas` (0064–0091) | Veredas con grietas o agujeros |
| `descarte` (0001–0033) | Escenas de interior |
| `descarte` (0034–0062) | Paisajes de montaña |
| `descarte` (0063–0095) | Mascotas (perros) |
| `descarte` (0096–fin) | Calles en buen estado, sin problemas visibles |

Cada imagen quedó asociada a una `expected_category` (la categoría objetivo del sistema, o `Descarte`) y, dentro de los descartes, a un `content_group` que identifica el subtipo de escena (interior, montaña, mascota o calle sin problema), usado en el desglose de resultados de la Fase 4.

## A.2. Generación de los textos de acompañamiento

Cada imagen se combina con un texto sintético que simula la descripción que un ciudadano adjuntaría al reporte. Los textos se generaron de forma programática (`generar_dataset.py`, semilla fija = 42, reproducible) para cubrir sistemáticamente la variabilidad esperada en reportes reales, evitando que el motor de texto pudiera resolver la tarea por simple coincidencia de palabras clave con el mapa semántico.

**Estilo de redacción** (`text_style`), cuatro categorías:
- **Claro:** nombra explícitamente el problema con vocabulario típico (p. ej. *"el semáforo de la esquina está roto"*).
- **Indirecto:** describe la situación sin usar ninguna palabra ancla de su propia categoría (p. ej. *"las luces de la esquina están apagadas y los autos pasan sin frenar"* en vez de mencionar "semáforo").
- **Formal:** redacción de reclamo administrativo (p. ej. *"Se solicita atender el siguiente inconveniente: rotura de cañería principal con pérdida de agua potable"*).
- **Vago:** frases genéricas de queja sin ninguna información sobre la categoría del problema (p. ej. *"esto está así hace meses, arreglen por favor"*), usadas también como texto trampa en los descartes.

**Nivel de ruido tipográfico** (`noise_level`), cuatro niveles, aplicados sobre la base ya redactada:
- **Ninguno:** texto sin alteraciones.
- **Leve:** minúsculas, sin puntuación, sin tildes.
- **Medio:** además del nivel leve, sustitución de palabras frecuentes por abreviaturas de uso coloquial (`que`→`q`, `por`→`x`, `de`→`d`, `porque`→`xq`, `también`→`tb`, etc.) y ocasional eliminación de un carácter interno en alguna palabra.
- **Fuerte:** además del nivel medio, sustituciones fonéticas frecuentes en la escritura informal (`c`→`k`, `s`→`z` en contexto de "se-/sé-", `v`↔`b`), eliminación de letras mudas, alargamiento vocálico (`sucio`→`sucioo`), fusión ocasional de dos palabras, uso de mayúsculas sostenidas y agregado de signos de puntuación o emojis de énfasis.

Cada imagen válida recibe además, con distinta probabilidad según el estilo, un prefijo conversacional ("hola,", "buenas,") y un sufijo de contexto (calle, barrio, antigüedad del reclamo), para acercar la longitud y estructura de los textos a un reporte ciudadano real.

Los descartes reciben, en cambio, uno de tres tipos de texto: (a) una frase "trampa" tomada del vocabulario de una categoría válida (simulando un intento de inyección de texto engañoso sobre una imagen irrelevante), (b) una frase vaga, o (c) un texto irrelevante (spam, saludos, consultas ajenas al sistema) o, para el subtipo "calle sin problema", una frase que celebra el buen estado de la vía pública.

Los 714 textos resultantes son todos distintos entre sí (no hay dos reportes con la misma descripción).

## A.3. Partición dev/test

El dataset incluye un campo `split` (dev/test), asignado de forma estratificada por `content_group`, de manera que ambas particiones contengan una proporción equivalente de cada subtipo de imagen. Las plantillas de texto base también se reparten sin superposición entre dev y test, de modo que una misma frase base no aparezca (en sus distintas variantes de ruido) en ambos lados de la partición.

## A.4. Campos del dataset

Cada elemento del archivo `real_dataset.json` contiene, entre otros: `id`, `image_path`, `expected_category`, `subset` (válido/descarte), `description` (el texto final, con estilo y ruido aplicados), `text_style`, `noise_level`, `content_group` y `split`.

## A.5. Alcance y limitaciones de este dataset

Este dataset fue diseñado para someter al sistema a una variabilidad textual controlada y medible (estilo × ruido), algo que un conjunto de reportes ciudadanos reales, recolectado sin este control, no garantiza por sí solo. Como contrapartida, las imágenes provienen de búsquedas web y tienden a ser más nítidas y prototípicas que una fotografía ciudadana tomada con un teléfono en condiciones reales (de noche, en movimiento o con mala iluminación), y los textos, aunque variados, son de autoría única y no reflejan necesariamente todos los patrones de redacción de la población de usuarios real. La validación del sistema sobre un conjunto de reportes ciudadanos reales, una vez disponible, es la extensión natural de este trabajo.
