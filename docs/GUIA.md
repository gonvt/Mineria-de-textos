# Guía del proyecto (para estudiar y defender la práctica)

Este documento explica **qué hace cada parte, por qué se tomó cada decisión y
dónde está el código**. El enunciado exige que el equipo pueda explicar y
modificar cualquier parte en la defensa oral: conviene leerlo junto al cuaderno.

---

## 1. Fase 1 – Dataset (`eonlp/dataset.py`)

### Temática

**Deportes**: `teniso` (tenis), `golfo` (golf) y `rugbeo` (rugby). Son cercanos
(jugadores, torneos, campeones, pelota, puntos, olimpiadas...), así que el
problema no es trivial.

**¿Por qué no pádel?** La elección inicial era pádel, golf y rugby, pero la
Vikipedio en Esperanto solo tiene un artículo de pádel de 4 frases
(*Padelo (sporto)*) y no hay otras fuentes en Esperanto. Con ~5 párrafos no se
puede entrenar una clase, así que se sustituyó por tenis, el deporte más
parecido. El artículo de pádel se clasifica al final como experimento.
**Conviene comentarlo con el profesor**, que es quien valida las categorías.

### Qué hace

| Función | Qué hace |
|---|---|
| `api_get` | Petición GET al API de Vikipedio con **4 reintentos** y espera creciente. Nunca lanza excepción: devuelve `None` si falla. |
| `category_members` | Lista páginas (`cmnamespace=0`) o subcategorías (`14`) de una categoría, siguiendo `cmcontinue`. |
| `collect_titles` | **Recorrido en anchura (BFS)** de la categoría del deporte y sus subcategorías (jugadores, torneos...) hasta 3 niveles. |
| `search_titles` | **Buscador** del API (`list=search`): artículos cuyo texto menciona el deporte. Hace falta porque las categorías son pequeñas (golf y rugby tienen menos de 20 artículos). |
| `fetch_extract` | Texto plano del artículo (`prop=extracts&explaintext=1`): el API ya quita HTML, plantillas y enlaces. |
| `split_paragraphs` | Corta por saltos de línea, ignora títulos `== ... ==` y **se detiene** en *Vidu ankaŭ, Referencoj, Notoj, Eksteraj ligiloj...* |
| `clean_paragraph` | Quita URLs, referencias `[1]`, LaTeX, cambia `;` por `,`; descarta párrafos de < 15 palabras o con < 75 % de letras; recorta a 300 palabras. |
| `build_dataset` | Aplica las reglas de abajo, quita duplicados y devuelve `text;class` (+ `title` y `source`). |
| `save_dataset` | Escribe el CSV con el contrato: UTF-8, `;`, cabecera `text;class`. |

### Decisiones que hay que saber justificar

* **Dos fuentes, dos reglas.** Artículo de la **categoría** → se guardan todos
  sus párrafos (heredan la etiqueta, con ruido). Artículo del **buscador** →
  solo los párrafos que contienen una **palabra clave** del deporte (`TOPICS`):
  de un artículo sobre los Juegos Olímpicos de 1920 solo interesa el párrafo
  del torneo de rugby.
* **"Golfo" es ambiguo**: en Esperanto también es un golfo geográfico (*Golfo
  de Meksiko*). Las palabras clave de golf son solo derivados inequívocos
  (*golfludo, golfejo, golfisto, golfbastono...*). Hay un test para esto.
* **Párrafos con dos deportes** (p. ej. resúmenes de olimpiadas) → se
  descartan por ambiguos.
* **Subcategorías excluidas**: *Usona piedpilko* y *NFL* cuelgan de *Rugbeo*
  en Vikipedio pero son fútbol americano.
* **Páginas reservadas** (`HOLDOUT_TITLES`): *Golfejo*, *Rugbea Monda Pokalo* y
  *Padelo (sporto)* se excluyen del dataset para usarlas en la prueba web.
* **¿Por qué se guarda el título?** Para la partición por artículo en la
  Fase 3. El contrato (`data/dataset.csv`) solo tiene `text;class`; el título y
  la fuente van en `data/dataset_con_titulos.csv`.
* **Ejecución asíncrona**: generar el dataset son ~1 150 peticiones (≈15 min).
  Por eso el CSV ya generado está en `data/` y el cuaderno lo descarga si
  `GENERAR_DATASET = False`. El código de generación sigue en el cuaderno.
* **Riesgo de la palabra clave**: como los párrafos del buscador se eligieron
  por contener *tenis-/golf-/rugbe-*, el modelo podría limitarse a buscar esa
  palabra. La sección 3.6 del cuaderno lo mide quitándolas.

---

## 2. Fase 2 – Modelo de NLP (`eonlp/nlp.py`, `eonlp/lexicon.py`)

### Idea principal

En Esperanto la categoría gramatical se ve en la **terminación**, sin
excepciones. Así que el etiquetador funciona en dos pasos:

1. ¿Está la palabra en el **léxico cerrado**? (`lexicon.py`: artículo `la`,
   pronombres `mi vi li ŝi ĝi ni ili oni si` y sus formas `-n`/`-a`, la tabla de
   45 **correlativos** `ki-/ti-/i-/ĉi-/neni-` × `-u -o -a -e -am -el -om -al -es`,
   preposiciones, conjunciones, adverbios primarios, numerales, interjecciones).
2. Si no, se aplican las **reglas de terminación** (`_by_ending`):
   `-o(j)(n)` sustantivo, `-a(j)(n)` adjetivo, `-e(n)` adverbio,
   `-as/-is/-os/-us/-u/-i` verbo. El lema se forma con la raíz + `-o`, `-a`,
   `-e` o `-i`.

### Casos especiales

* **Participios** (`-ant-, -int-, -ont-, -at-, -it-, -ot-`): *skribita* →
  lema *skribi*, con `VerbForm=Part`, aspecto y voz. Hay palabras que parecen
  participios y no lo son (*konstanta, elefanta, privata*): lista
  `PARTICIPLE_EXCEPTIONS`.
* **Nombres propios**: mayúscula en mitad de la oración → `PROPN`. A inicio de
  oración no se puede saber, así que se aplican las reglas normales.
* **Palabras extranjeras**: letras que no existen en Esperanto (`q w x y`) →
  `PROPN` si van en mayúscula, `X` si no.
* **Numerales compuestos** (*dudek, tricent kvin*): expresión regular `_NUM_RE`;
  ordinales (*dua, tria*) → `ADJ` con `NumType=Ord`.
* **Sistema‑x** (`cx` → `ĉ`): se normaliza antes de tokenizar.
* **Variantes con ĥ** (*ĥemio* → *kemio*): `unify_spelling`. En Vikipedio
  conviven las dos grafías y sin esto el modelo vería dos palabras distintas.
* **Morfemas** (`split_morphemes`): quita prefijos y sufijos oficiales mientras
  la raíz conserve ≥ 3 letras: *malsanulejo* → `mal + san + ul + ej`.

### API tipo spaCy

| spaCy | Aquí |
|---|---|
| `spacy.load(...)` | `eonlp.load()` |
| `nlp(text)` → `Doc` | igual |
| `nlp.pipe(texts)` | igual (generador) |
| `nlp.pipe_names` | `['normalizer', 'tokenizer', 'tagger', 'lemmatizer', 'stopwords']` |
| `token.text, .i, .idx, .whitespace_` | igual |
| `token.lemma_, .pos_, .tag_, .morph` | igual (`morph` imprime `Case=Acc\|Number=Plur`) |
| `token.is_stop, .is_punct, .is_alpha, .like_num` | igual |
| `doc.sents`, `doc[2:5]` → `Span` | igual |
| — | extra: `token.root_` (raíz) y `token.morphemes` |

### Stop-words

`default_stop_words()` = todas las palabras del léxico cerrado + verbos
auxiliares (*esti, havi, povi, devi, voli, fari...*) + palabras muy frecuentes
sin tema en una enciclopedia (*jaro, parto, ekzemplo, granda, nova...*).
Se comparan por **lema**, así *estas/estis/estos* son stop-words por *esti*.

---

## 3. Fase 3 – Clasificación (`eonlp/classify.py`)

* `doc_to_features(doc, mode)`: se quedan solo las palabras con contenido
  (`NOUN, PROPN, ADJ, VERB, ADV`), sin stop-words ni números, y se representan
  por su `text`, `lemma` o `root`.
* `make_vectorizer`: **TF-IDF** con `sublinear_tf=True` (log de la frecuencia),
  `min_df=2` (quita palabras que salen una sola vez) y `max_df=0.9`.
* `candidate_models`: Naive Bayes multinomial, regresión logística y SVM lineal.
* **Desbalanceo** (966 tenis / 328 rugby / 218 golf): NB con `fit_prior=False`
  y los modelos lineales con `class_weight="balanced"`; métrica principal F1 macro.
* **Evaluación**: `StratifiedGroupKFold` agrupando por **artículo**. Con una
  validación aleatoria el F1 sube de 0.84 a 0.92, pero es engañoso: párrafos del
  mismo artículo en train y test comparten nombres y términos concretos.
* Se elige la mejor combinación (SVM lineal + raíces, `C=1`, sin bigramas). Como
  la SVM no da probabilidades, se calibra con `CalibratedClassifierCV`. En test
  (≈25 % de los artículos) obtiene accuracy 0.88 y F1 macro 0.87.
* **Sección 3.6**: sin las palabras *tenis-/golf-/rugbe-* el F1 cae a 0.66. Es
  la principal debilidad: el modelo se apoya en el nombre del deporte.

### Clasificación de páginas web

* `fetch_html` + `html_to_paragraphs`: `requests` + `BeautifulSoup`; se quitan
  `script, style, nav, table, sup...` y las cajas de Wikipedia, y se extraen los
  `<p>`, limpiados con la misma función que el dataset.
* Páginas: una noticia de **Vikinovaĵoj** (tenis), *Golfejo* y *Rugbea Monda
  Pokalo* (reservadas, fuera del dataset) y *Padelo (sporto)*. Todas aciertan;
  el pádel sale como tenis.
* Probabilidad de la página = **media de las probabilidades de sus párrafos**.
* Si la web falla, se usa `data/web_snapshot.json` (copia de respaldo); si
  también falla, se omite la página con un aviso. El cuaderno nunca se rompe.

---

## 4. Preguntas probables en la defensa

* *¿Cómo lematizas "pilkojn"?* → no está en el léxico; termina en `-ojn` →
  sustantivo, plural, acusativo; lema = raíz `pilk` + `o` = *pilko*.
* *¿Qué pasa con "konstanta"?* → parece participio (`-ant-a`) pero está en la
  lista de excepciones, así que su lema es *konstanta*.
* *¿Por qué no hay pádel?* → solo hay un artículo de 4 frases en Esperanto; se
  cambió por tenis y el artículo de pádel se clasifica como prueba.
* *¿Por qué golf tiene menos de 300 párrafos?* → es todo lo que hay en la
  Vikipedio en Esperanto; se compensa con F1 macro y pesos de clase.
* *¿Por qué "golfo" no es una palabra clave?* → también significa golfo
  geográfico; se usan derivados (*golfludo, golfejo, golfisto*).
* *¿El modelo solo mira el nombre del deporte?* → en parte sí (sección 3.6:
  F1 0.84 → 0.66 sin esas palabras).
* *Cambia las categorías / añade una stop-word / usa otro modelo* → `TOPICS`
  en la Fase 1, `nlp.add_stop_words([...])`, `candidate_models()`.
