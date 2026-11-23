# Guía del proyecto (para estudiar y defender la práctica)

Este documento explica **qué hace cada parte, por qué se tomó cada decisión y
dónde está el código**. El enunciado exige que el equipo pueda explicar y
modificar cualquier parte en la defensa oral: conviene leerlo junto al cuaderno.

---

## 1. Fase 1 – Dataset (`eonlp/dataset.py`)

### Qué hace

| Función | Qué hace |
|---|---|
| `api_get` | Petición GET al API de Vikipedio con **4 reintentos** y espera creciente. Nunca lanza excepción: devuelve `None` si falla. |
| `category_members` | Lista páginas (`cmnamespace=0`) o subcategorías (`14`) de una categoría, siguiendo `cmcontinue` si hay más de 500. |
| `collect_titles` | **Recorrido en anchura (BFS)**: primero los artículos de la categoría raíz, luego los de sus subcategorías, hasta 2 niveles o 300 artículos. Ignora títulos que empiezan por *Listo* (listas). |
| `fetch_extract` | Texto plano del artículo (`prop=extracts&explaintext=1`): el API ya quita HTML, plantillas y enlaces. |
| `split_paragraphs` | Corta por saltos de línea, ignora títulos `== ... ==` y **se detiene** en *Vidu ankaŭ, Referencoj, Notoj, Eksteraj ligiloj...* |
| `clean_paragraph` | Quita URLs, referencias `[1]`, LaTeX, cambia `;` por `,`; descarta párrafos de < 15 palabras o con < 75 % de letras; recorta a 300 palabras. |
| `build_dataset` | Junta todo, descarta artículos que estén en **dos clases** y párrafos **duplicados**. |
| `save_dataset` | Escribe el CSV con el contrato: UTF-8, `;`, cabecera `text;class`. |

### Decisiones que hay que saber justificar

* **¿Por qué física, química y biología?** Son cercanas (comparten
  *energio, atomo, molekulo, ĉelo, reakcio*), así que el problema no es trivial,
  y Vikipedio tiene suficientes artículos de las tres.
* **¿Por qué se excluyen subcategorías?** `EXCLUDED_SUBCATEGORIES` quita
  biografías (*-istoj*, *-ologoj*), listas, organizaciones, premios, esbozos
  (*Ĝermoj*) y las interdisciplinares (*Biofiziko, Biokemio, Kvantuma kemio*):
  sus párrafos serían de dos clases a la vez. *Metalurgio* y *Kemia industrio*
  se quitaron tras ver que el BFS acababa en artículos de fábricas de acero.
* **¿Por qué se guarda el título?** Para la partición por artículo en la
  Fase 3. El contrato (`data/dataset.csv`) solo tiene `text;class`; el título
  va en un fichero aparte (`data/dataset_con_titulos.csv`).
* **Ejecución asíncrona**: generar el dataset son ~900 peticiones (10–15 min).
  Por eso el CSV ya generado está publicado en `data/` y el cuaderno lo
  descarga si `GENERAR_DATASET = False`. El código de generación sigue estando
  en el cuaderno (requisito del enunciado).
* **Ruido**: cada párrafo hereda la etiqueta del artículo. Se mide de dos formas
  (sección 1.5 y 3.5 del cuaderno): párrafos sin vocabulario de dominio
  (≈23 %, medida gruesa) y párrafos que el clasificador contradice con
  probabilidad > 0.8 en validación cruzada (≈5.6 %). Entrenar sin esos
  párrafos apenas cambia el F1 (0.850 → 0.852).

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
* **Evaluación**: `StratifiedGroupKFold` agrupando por **artículo**. Con una
  validación aleatoria el F1 sube de 0.81 a 0.89, pero es engañoso: párrafos del
  mismo artículo en train y test comparten nombres y términos concretos.
* Se elige la mejor combinación (Naive Bayes + raíces, con bigramas,
  `alpha=0.1`), se evalúa en un test del 20 % de artículos (accuracy 0.85,
  F1 macro 0.85) y se reentrena con todo para clasificar las webs.

### Clasificación de páginas web

* `fetch_html` + `html_to_paragraphs`: `requests` + `BeautifulSoup`; se quitan
  `script, style, nav, table, sup...` y las cajas de Wikipedia, y se extraen los
  `<p>`, limpiados con la misma función que el dataset.
* Probabilidad de la página = **media de las probabilidades de sus párrafos**
  (clasificar la página entera de golpe satura Naive Bayes a 0 %/100 %).
* Si la web falla, se usa `data/web_snapshot.json` (copia de respaldo); si
  también falla, se omite la página con un aviso. El cuaderno nunca se rompe.

---

## 4. Preguntas probables en la defensa

* *¿Cómo lematizas "belajn"?* → no está en el léxico; termina en `-ajn` →
  adjetivo, plural, acusativo; lema = raíz `bel` + `a` = *bela*.
* *¿Qué pasa con "konstanta"?* → parece participio (`-ant-a`) pero está en la
  lista de excepciones, así que su lema es *konstanta*.
* *¿Por qué Naive Bayes gana a la SVM?* → con textos cortos y muchas
  palabras raras, NB generaliza mejor con pocos datos; además da probabilidades
  directamente. Las diferencias son pequeñas (≈2 puntos).
* *¿Por qué Fotosintezo sale como química?* → la mayoría de sus párrafos son
  de reacciones y moléculas (bioquímica) y excluimos *Biokemio* del dataset.
* *Cambia las categorías / añade una stop-word / usa otro modelo* → `CATEGORIES`
  en la Fase 1, `nlp.add_stop_words([...])`, `candidate_models()`.
