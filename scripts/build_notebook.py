"""Genera el cuaderno ``notebooks/clasificacion_esperanto.ipynb``.

El enunciado exige que todo el código (descarga del dataset, modelo de NLP y
clasificador) esté DENTRO del cuaderno. Para no tener el código duplicado a
mano, la fuente única es el paquete ``eonlp/`` y este script copia el código
de cada módulo en las celdas del cuaderno (quitando los ``import`` relativos).

Uso::

    python scripts/build_notebook.py            # solo genera el .ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/clasificacion_esperanto.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "eonlp"
OUT = ROOT / "notebooks" / "clasificacion_esperanto.ipynb"
REPO = "gonvt/Mineria-de-textos"
import os

BRANCH = os.environ.get("NB_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"


def module_source(name):
    """Código del módulo sin los imports relativos (en el cuaderno todo es global)."""
    lines = (PKG / f"{name}.py").read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.startswith("from .")]
    return f"# ===== Código de eonlp/{name}.py =====\n" + "\n".join(kept).strip() + "\n"


cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ==========================================================================
# PORTADA
# ==========================================================================
md(f"""
# Clasificación temática de textos en Esperanto
### Práctica de programación · Minería de Textos · 4º INF · 2026/2027

**Integrantes:** _(completar con los nombres del equipo)_

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/clasificacion_esperanto.ipynb)

En este cuaderno se construye, de principio a fin, un sistema que clasifica
textos en **Esperanto** según su temática científica:

| Clase | Categoría de Vikipedio | Qué contiene |
|---|---|---|
| `fiziko` | [Kategorio:Fiziko](https://eo.wikipedia.org/wiki/Kategorio:Fiziko) | física: mecánica, óptica, electromagnetismo... |
| `kemio` | [Kategorio:Kemio](https://eo.wikipedia.org/wiki/Kategorio:Kemio) | química: elementos, reacciones, compuestos... |
| `biologio` | [Kategorio:Biologio](https://eo.wikipedia.org/wiki/Kategorio:Biologio) | biología: células, ecología, genética... |

Las tres son ramas de las ciencias naturales, **semánticamente cercanas**
(comparten vocabulario: *energio, molekulo, ĉelo, atomo, reakcio*...), así que
el problema no es trivial.

**Índice**

0. Configuración y parámetros
1. **Fase 1 – Dataset**: descarga de Vikipedio, limpieza y división en párrafos
2. **Fase 2 – NLP**: modelo propio para Esperanto con API tipo spaCy
3. **Fase 3 – Clasificación**: vectorización TF-IDF, entrenamiento y evaluación
4. **Clasificación de páginas web** reales mediante *web scraping*
5. Conclusiones

> El código de las fases 1–3 también está organizado como librería instalable
> (`pip install git+https://github.com/{REPO}`) con tests, en la carpeta `eonlp/`
> del repositorio. Este cuaderno es **autocontenido**: no importa esa librería,
> sino que incluye todo su código en las celdas.
""")

# ==========================================================================
# 0. CONFIGURACIÓN
# ==========================================================================
md("""
## 0. Configuración y parámetros

Todos los parámetros de ejecución están en esta celda.

* `GENERAR_DATASET = False` (por defecto): se usa el dataset ya generado y
  publicado en GitHub (*ejecución asíncrona*, sección 4.1.4 del enunciado).
  Así el cuaderno tarda menos de 2 minutos.
* `GENERAR_DATASET = True`: se ejecuta el *script* de la Fase 1 y se descarga
  todo de nuevo desde el API de Vikipedio (unos 10–15 minutos, ~900 peticiones).

Si la descarga del dataset publicado falla, el cuaderno lo genera
automáticamente desde Vikipedio (y viceversa): **sobrevive al fallo de cualquiera
de las dos fuentes**.
""")
code(f"""
# ---------------------------- PARÁMETROS ----------------------------------
GENERAR_DATASET = False        # True = volver a descargar todo desde Vikipedio
MAX_ARTICULOS_POR_CLASE = 300  # artículos por categoría (solo si se genera)
PROFUNDIDAD_SUBCATEGORIAS = 2  # niveles de subcategorías a recorrer
SEED = 42                      # semilla para que los resultados sean reproducibles

URL_DATASET = "{RAW}/data/dataset_con_titulos.csv"   # text;class;title
URL_DATASET_CONTRATO = "{RAW}/data/dataset.csv"      # text;class (contrato de datos)
URL_WEB_SNAPSHOT = "{RAW}/data/web_snapshot.json"     # copia de respaldo de las webs
""")
code("""
import json
import random
import re
import sys
import time
import unicodedata
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(SEED)
np.random.seed(SEED)

pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLORES = {"fiziko": "#2a78b5", "kemio": "#d9822b", "biologio": "#3a9a5b"}

print("Python      ", sys.version.split()[0])
for lib in (pd, np, sklearn, matplotlib, requests):
    print(f"{lib.__name__:<12}", lib.__version__)
""")

# ==========================================================================
# FASE 1
# ==========================================================================
md("""
## 1. Fase 1 – Dataset

### 1.1 Fuente de datos y temáticas

Se usa el **API de MediaWiki de Vikipedio** (`eo.wikipedia.org/w/api.php`):

1. `list=categorymembers` para recorrer cada categoría **en anchura (BFS)** y sus
   subcategorías hasta 2 niveles, hasta reunir 300 artículos por clase.
2. `prop=extracts&explaintext=1` para obtener el **texto plano** de cada artículo
   (sin HTML, plantillas ni enlaces).

**Decisiones de diseño**

* **Subcategorías excluidas**: biografías (*Fizikistoj, Kemiistoj, Biologoj*),
  listas, organizaciones, premios, esbozos (*Ĝermoj*) y, sobre todo, las
  **interdisciplinares** (*Biofiziko, Biokemio, Kvantuma kemio*), que
  pertenecerían a dos clases a la vez. También *Metalurgio* / *Kemia industrio*,
  que se alejan hacia la industria y la historia.
* Un artículo que aparece en **más de una clase** se descarta (etiqueta ambigua).
* Cada artículo se divide en **párrafos**; se ignoran los títulos de sección y
  todo lo que va después de *Vidu ankaŭ, Referencoj, Notoj, Eksteraj ligiloj*...
* Limpieza: URLs, referencias `[1]`, fórmulas LaTeX, el carácter `;`
  (contrato de datos) y párrafos con menos de 15 palabras o con menos de un
  75 % de letras (tablas, listas de cifras).
* Se eliminan los párrafos duplicados (textos de plantilla repetidos).
* Las peticiones tienen **reintentos con espera** y nunca lanzan excepción:
  si una petición falla, ese artículo simplemente se queda sin texto.
""")
code(module_source("dataset"))

md("""
### 1.2 Generar o cargar el dataset

Se guarda además el **título del artículo** de cada párrafo. No forma parte del
contrato de datos (`text;class`), pero es imprescindible para evaluar bien: los
párrafos de un mismo artículo se parecen mucho entre sí, así que en la Fase 3 se
separa train/test **por artículo** para no "hacer trampa".
""")
code("""
def generar_dataset():
    print("Descargando artículos de Vikipedio... (puede tardar ~10-15 min)")
    t0 = time.time()
    raw = download_articles(CATEGORIES, MAX_ARTICULOS_POR_CLASE, PROFUNDIDAD_SUBCATEGORIAS)
    df = build_dataset(raw)
    print(f"Dataset generado en {time.time() - t0:.0f} s")
    return df


def cargar_dataset_publicado():
    print("Cargando el dataset publicado en GitHub...")
    return pd.read_csv(URL_DATASET, sep=";", encoding="utf-8")


fuentes = [generar_dataset, cargar_dataset_publicado]
if not GENERAR_DATASET:
    fuentes.reverse()

df = None
for fuente in fuentes:
    try:
        df = fuente()
        if df is not None and len(df) > 0:
            break
    except Exception as exc:  # el cuaderno debe sobrevivir a fallos de red
        print(f"  [aviso] {fuente.__name__} falló: {type(exc).__name__}: {exc}")
if df is None or len(df) == 0:
    raise RuntimeError("No se pudo obtener el dataset de ninguna fuente.")

# Guardar el dataset con el contrato de datos (útil para subirlo a GitHub)
save_dataset(df, "dataset.csv")
print(f"\\n{len(df)} párrafos · {df['title'].nunique()} artículos · guardado en dataset.csv")
df.head()
""")

md("### 1.3 Verificación del contrato de datos")
code("""
contrato = pd.read_csv("dataset.csv", sep=";", encoding="utf-8")
comprobaciones = {
    "cabecera exacta 'text;class'": list(contrato.columns) == ["text", "class"],
    "sin ';' en los párrafos": not contrato["text"].str.contains(";").any(),
    "sin HTML": not contrato["text"].str.contains(r"<[a-zA-Z/][^>]*>").any(),
    "sin URLs": not contrato["text"].str.contains(r"https?://|www\\.").any(),
    "sin referencias [n]": not contrato["text"].str.contains(r"\\[\\d+\\]").any(),
    "sin valores vacíos": not contrato.isna().any().any(),
    ">= 300 párrafos por clase": (contrato["class"].value_counts() >= 300).all(),
}
for regla, ok in comprobaciones.items():
    print(("✅" if ok else "❌"), regla)
assert all(comprobaciones.values())
""")

md("### 1.4 Exploración del dataset")
code("""
resumen = df.groupby("class").agg(
    parrafos=("text", "size"),
    articulos=("title", "nunique"),
    palabras_media=("text", lambda s: s.str.split().str.len().mean()),
).round(1).loc[list(CATEGORIES)]
display(resumen)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
resumen["parrafos"].plot.bar(ax=axes[0], color=[COLORES[c] for c in resumen.index], rot=0)
axes[0].set_title("Párrafos por clase")
for c in CATEGORIES:
    df.loc[df["class"] == c, "text"].str.split().str.len().clip(upper=200).plot.hist(
        ax=axes[1], bins=40, alpha=0.55, color=COLORES[c], label=c)
axes[1].set_title("Longitud de los párrafos (palabras)"); axes[1].legend()
df.groupby("title").size().clip(upper=40).plot.hist(ax=axes[2], bins=40, color="grey")
axes[2].set_title("Párrafos por artículo")
plt.tight_layout(); plt.show()
""")
md("""
El dataset está algo **desbalanceado** (física tiene más párrafos porque sus
artículos son más largos). No se re-muestrea: se usará **F1 macro** como métrica
principal, que pesa igual las tres clases. También hay artículos muy largos
(más de 100 párrafos) que pesan mucho en su clase; por eso la partición
train/test se hace por artículo.
""")
code("""
for c in CATEGORIES:
    ejemplo = df[df["class"] == c].sample(1, random_state=SEED).iloc[0]
    print(f"[{c}] ({ejemplo['title']})\\n  {ejemplo['text'][:300]}\\n")
""")

md("""
### 1.5 Ruido de etiquetado (primera aproximación)

Cada párrafo **hereda la etiqueta de su artículo**, pero muchos párrafos son
genéricos: fechas, historia, biografías... (*"Ĝi estis fondita en 1950"*).
Una medida sencilla del ruido: buscar párrafos que **no contienen ningún término
del dominio**. Se define el vocabulario de dominio con raíces típicas de cada
ciencia (lista propia, sin usar el modelo, para no tener un razonamiento circular).
En la Fase 3 se mide el ruido de una segunda forma, con el propio clasificador.
""")
code("""
RAICES_DOMINIO = {
    "fiziko": ["fizik", "energi", "fort", "mas", "rapid", "ond", "lum", "elektr", "magnet",
               "kvantum", "partikl", "gravit", "temperatur", "radiad", "mekanik", "optik",
               "frekvenc", "akcel", "kamp", "relativ"],
    "kemio": ["kemi", "ĥemi", "atom", "molekul", "element", "reakci", "acid", "baz", "jon",
              "metal", "oksid", "kombinaĵ", "solv", "hidrogen", "oksigen", "karbon", "salo",
              "izotop", "kristal", "gas"],
    "biologio": ["biolog", "ĉel", "organism", "speci", "gen", "protein", "plant", "best",
                 "evoluci", "evolu", "histo", "ekolog", "bakteri", "vivaĵ", "hormon",
                 "kromosom", "dna", "sang", "nutr", "reprodukt"],
}
todas = [r for rs in RAICES_DOMINIO.values() for r in rs]
patron = re.compile(r"\\b(" + "|".join(todas) + r")", re.IGNORECASE)
df["sin_termino_de_dominio"] = ~df["text"].str.contains(patron)

ruido_heuristico = df.groupby("class")["sin_termino_de_dominio"].mean().mul(100).round(1)
print("% de párrafos sin ningún término científico:")
print(ruido_heuristico.to_string(), f"\\nTotal: {df['sin_termino_de_dominio'].mean()*100:.1f} %")
print("\\nEjemplos:")
for t in df[df["sin_termino_de_dominio"]].sample(4, random_state=SEED)["text"]:
    print(" ·", t[:160])
""")

# ==========================================================================
# FASE 2
# ==========================================================================
md("""
## 2. Fase 2 – Modelo de NLP para Esperanto

Se implementa **desde cero** (sin spaCy, NLTK ni Stanza) un modelo basado en
reglas con un API como el de spaCy:

```python
nlp = load()
doc = nlp("La katoj manĝis fiŝon.")
for token in doc:
    token.text, token.lemma_, token.pos_, token.morph, token.is_stop
```

**¿Por qué funcionan las reglas?** El Esperanto marca la categoría gramatical
con la terminación, sin excepciones:

| Terminación | Categoría | Rasgos | Ejemplo → lema |
|---|---|---|---|
| `-o` (`-oj`, `-on`, `-ojn`) | sustantivo `NOUN` | número, caso | *atomojn* → *atomo* |
| `-a` (`-aj`, `-an`, `-ajn`) | adjetivo `ADJ` | número, caso | *kemiajn* → *kemia* |
| `-e` (`-en`) | adverbio `ADV` | dirección | *rapide* → *rapide* |
| `-as` / `-is` / `-os` | verbo `VERB` | presente / pasado / futuro | *trovis* → *trovi* |
| `-us` / `-u` / `-i` | verbo `VERB` | condicional / imperativo / infinitivo | *estu* → *esti* |
| `-ant- -int- -ont- -at- -it- -ot-` | participio | aspecto, voz | *skribita* → *skribi* |

Las palabras que no siguen estas reglas son de **clase cerrada** (artículo,
pronombres, la tabla de 45 correlativos, preposiciones, conjunciones,
numerales...) y se buscan en un **léxico** antes de aplicar las reglas.

**Pipeline** (`nlp.pipe_names`):

1. `normalizer`: Unicode NFC, apóstrofos y **sistema‑x** (`cx` → `ĉ`, `sx` → `ŝ`).
2. `tokenizer`: expresión regular (palabras con guion y elisión `l'`, números, URLs, signos).
3. `tagger`: UPOS + rasgos morfológicos (Universal Dependencies) por léxico y terminaciones.
   Mayúscula en mitad de oración → `PROPN`; letras `q w x y` → palabra extranjera.
4. `lemmatizer`: forma de diccionario y **descomposición en morfemas**
   (`malsanulejo` → `mal + san + ul + ej + o`) con prefijos/sufijos oficiales.
5. `stopwords`: `token.is_stop` a partir de una lista de **lemas** (así *estas,
   estis, estos* son todas stop-word por el lema *esti*).

También unifica variantes ortográficas con *ĥ* (*ĥemio* → *kemio*), muy
frecuentes en Vikipedio.
""")
code(module_source("lexicon"))
code(module_source("nlp"))

md("### 2.1 Demostración del API")
code("""
nlp = load()
print(nlp)
doc = nlp("La junaj sciencistoj malkovris novajn elementojn en la laboratorio de Marie Curie, "
          "kaj ili skribos artikolon pri la radioaktiveco.")
pd.DataFrame(doc.to_table())
""")
code("""
# Oraciones, morfemas, sistema-x y stop-words
doc = nlp("Mi sxatas la malsanulejon. Cxu vi vidis la ĉielarkon hieraŭ?")
print("Oraciones:", [s.text for s in doc.sents])
print("Normalizado (sistema-x):", doc.text)
for palabra in ["malsanulejo", "ĉielarko", "lernejestro", "maljuneco", "fotosintezo"]:
    print(f"  {palabra:<12} → morfemas {nlp(palabra)[0].morphemes}")
print("Stop-words en el texto:", [t.text for t in doc if t.is_stop])
print("Tamaño de la lista de stop-words:", len(nlp.stop_words))
""")

md("""
### 2.2 Evaluación del modelo sobre un pequeño *gold standard*

Para comprobar que el etiquetador funciona se anotan a mano unas frases
(etiqueta UPOS y lema esperados) y se mide la exactitud.
""")
code("""
GOLD = [
    ("La katoj manĝis fiŝon .", "DET NOUN VERB NOUN PUNCT", "la kato manĝi fiŝo ."),
    ("Mi ŝatas legi interesajn librojn .", "PRON VERB VERB ADJ NOUN PUNCT", "mi ŝati legi interesa libro ."),
    ("Kiuj sciencistoj malkovris la elektronon ?", "DET NOUN VERB DET NOUN PUNCT", "kiu sciencisto malkovri la elektrono ?"),
    ("La akvo bolas je cent gradoj .", "DET NOUN VERB ADP NUM NOUN PUNCT", "la akvo boli je cent grado ."),
    ("Ŝi iras hejmen rapide ĉar pluvas .", "PRON VERB ADV ADV SCONJ VERB PUNCT", "ŝi iri hejme rapide ĉar pluvi ."),
    ("Ĉiuj vivaj ĉeloj enhavas DNA kaj proteinojn .", "DET ADJ NOUN VERB PROPN CCONJ NOUN PUNCT", "ĉiu viva ĉelo enhavi DNA kaj proteino ."),
    ("La skribita letero estis sendita al Parizo .", "DET ADJ NOUN VERB ADJ ADP PROPN PUNCT", "la skribi letero esti sendi al Parizo ."),
    ("Se vi volus , ni povus helpi vin .", "SCONJ PRON VERB PUNCT PRON VERB VERB PRON PUNCT", "se vi voli , ni povi helpi vi ."),
    ("Tiu ĉi reakcio produktas varmon kaj lumon .", "DET PART NOUN VERB NOUN CCONJ NOUN PUNCT", "tiu ĉi reakcio produkti varmo kaj lumo ."),
    ("La dua leĝo de Newton priskribas la forton .", "DET ADJ NOUN ADP PROPN VERB DET NOUN PUNCT", "la dua leĝo de Newton priskribi la forto ."),
]
aciertos_pos = aciertos_lema = total = 0
errores = []
for frase, pos, lemas in GOLD:
    doc = nlp(frase)
    for tok, p, l in zip(doc, pos.split(), lemas.split()):
        total += 1
        aciertos_pos += tok.pos_ == p
        aciertos_lema += tok.lemma_ == l
        if tok.pos_ != p or tok.lemma_ != l:
            errores.append((tok.text, tok.pos_, p, tok.lemma_, l))
print(f"Exactitud POS:   {aciertos_pos/total:.1%}  ({total} tokens)")
print(f"Exactitud lemas: {aciertos_lema/total:.1%}")
if errores:
    display(pd.DataFrame(errores, columns=["token", "pos_pred", "pos_gold", "lema_pred", "lema_gold"]))
""")
md("""
### 2.3 El modelo sobre el corpus

Distribución de categorías gramaticales en el dataset y palabras que el modelo
no sabe clasificar (`X`). Esto sirve para detectar casos que las reglas no
cubren (palabras extranjeras, fórmulas químicas, abreviaturas).
""")
code("""
t0 = time.time()
docs = list(nlp.pipe(df["text"]))
print(f"{len(docs)} párrafos procesados en {time.time() - t0:.1f} s "
      f"({sum(len(d) for d in docs):,} tokens)")

conteo_pos = pd.Series([t.pos_ for d in docs for t in d]).value_counts()
ax = (conteo_pos / conteo_pos.sum() * 100).plot.barh(figsize=(7, 4), color="#555")
ax.invert_yaxis(); ax.set_xlabel("% de tokens"); ax.set_title("Categorías gramaticales en el corpus")
plt.show()
desconocidas = pd.Series([t.text for d in docs for t in d if t.pos_ == "X"]).value_counts()
print("Palabras más frecuentes etiquetadas como X:", list(desconocidas.head(15).index))
""")

# ==========================================================================
# FASE 3
# ==========================================================================
md("""
## 3. Fase 3 – Clasificación

### 3.1 Representación tabular (vectorización)

Cada párrafo se convierte, con el modelo de la Fase 2, en la lista de
**lemas de las palabras con contenido** (sustantivos, nombres propios,
adjetivos, verbos y adverbios), sin stop-words, números ni signos. Después se
aplica **TF-IDF** (`sublinear_tf`, `min_df=2`, `max_df=0.9`).

Se comparan tres representaciones para justificar el uso del modelo de NLP:

* `text`: palabra en minúsculas, **sin lematizar** (línea base);
* `lemma`: **lema** (`atomojn`, `atomo` → `atomo`);
* `root`: **raíz** sin terminación (`fiziko`, `fizika`, `fizike` → `fizik`).
""")
code(module_source("classify"))
code("""
t0 = time.time()
# Reutilizamos los Doc ya procesados en la sección 2.3 (una sola pasada del modelo)
REPRESENTACIONES = {m: [" ".join(doc_to_features(d, m)) for d in docs] for m in ("text", "lemma", "root")}
print(f"Preprocesado en {time.time() - t0:.1f} s")
print("Original :", df['text'][0][:150])
print("lemma    :", REPRESENTACIONES['lemma'][0][:150])
print("root     :", REPRESENTACIONES['root'][0][:150])

for m, X_txt in REPRESENTACIONES.items():
    print(f"{m:<6} → vocabulario TF-IDF de {make_vectorizer().fit(X_txt).transform(X_txt).shape[1]:,} términos")
""")

md("""
### 3.2 Metodología de evaluación

* **Validación cruzada de 5 particiones agrupada por artículo**
  (`StratifiedGroupKFold`): todos los párrafos de un mismo artículo caen en la
  misma partición. Si se mezclan, el modelo "reconoce" el artículo (nombres
  propios, términos concretos) y la nota se infla artificialmente (se muestra
  abajo).
* Métricas: **accuracy** y **F1 macro** (media de las tres clases, robusta al
  desbalanceo).
* Con la mejor configuración se hace una partición **train/test 80/20 por
  artículo** y se dan la matriz de confusión y el informe por clase.
* Clasificadores ligeros: Naive Bayes multinomial, regresión logística y SVM
  lineal (todo se entrena en segundos sin GPU).
""")
code("""
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

y = df["class"].values
grupos = df["title"].values
cv_grupos = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

filas = []
t0 = time.time()
for rep, X_txt in REPRESENTACIONES.items():
    for nombre, modelo in candidate_models().items():
        r = cross_validate(build_pipeline(modelo), X_txt, y, groups=grupos, cv=cv_grupos,
                           scoring=["accuracy", "f1_macro"], n_jobs=-1)
        filas.append({"representación": rep, "modelo": nombre,
                      "accuracy": r["test_accuracy"].mean(), "f1_macro": r["test_f1_macro"].mean(),
                      "f1_std": r["test_f1_macro"].std()})
comparativa = pd.DataFrame(filas).sort_values("f1_macro", ascending=False).reset_index(drop=True)
print(f"Validación cruzada completada en {time.time() - t0:.0f} s")
display(comparativa.style.format({"accuracy": "{:.3f}", "f1_macro": "{:.3f}", "f1_std": "±{:.3f}"})
        .background_gradient(subset=["f1_macro"], cmap="Greens"))
""")
code("""
pivot = comparativa.pivot(index="modelo", columns="representación", values="f1_macro")[["text", "lemma", "root"]]
ax = pivot.plot.bar(figsize=(8, 3.8), rot=0, color=["#bbb", "#2a78b5", "#7fb3d5"])
ax.set_ylim(pivot.values.min() - 0.05, pivot.values.max() + 0.02)
ax.set_ylabel("F1 macro (CV por artículo)"); ax.set_title("Comparación de representaciones y modelos")
ax.legend(title="representación", loc="lower right"); plt.show()
""")

md("""
#### Ajuste de hiperparámetros

Sobre la mejor representación se ajusta el hiperparámetro principal del mejor
modelo y se prueba si añadir **bigramas** ayuda (p. ej. *nigra truo*, *kemia
reakcio*).
""")
code("""
from sklearn.model_selection import GridSearchCV

mejor = comparativa.iloc[0]
REP = mejor["representación"]
X_txt = REPRESENTACIONES[REP]
modelo_base = candidate_models()[mejor["modelo"]]
param = {"Naive Bayes": {"clf__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]},
         "Regresión logística": {"clf__C": [1, 3, 10, 30]},
         "SVM lineal": {"clf__C": [0.1, 0.3, 1, 3]}}[mejor["modelo"]]
param["tfidf__ngram_range"] = [(1, 1), (1, 2)]

busqueda = GridSearchCV(build_pipeline(modelo_base), param, scoring="f1_macro", cv=cv_grupos, n_jobs=-1)
busqueda.fit(X_txt, y, groups=grupos)
print(f"Modelo: {mejor['modelo']} · representación: {REP}")
print("Mejores hiperparámetros:", busqueda.best_params_, f"→ F1 macro CV = {busqueda.best_score_:.3f}")
""")

md("""
#### ¿Por qué agrupar por artículo? (fuga de información)

Misma configuración, evaluada con una validación cruzada **aleatoria** por párrafos:
""")
code("""
cv_aleatoria = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
f1_aleatoria = cross_validate(busqueda.best_estimator_, X_txt, y, cv=cv_aleatoria,
                              scoring="f1_macro", n_jobs=-1)["test_score"].mean()
print(f"F1 macro con CV aleatoria por párrafo : {f1_aleatoria:.3f}  (optimista)")
print(f"F1 macro con CV agrupada por artículo : {busqueda.best_score_:.3f}  (realista)")
""")

md("### 3.3 Modelo final: entrenamiento y evaluación en test")
code("""
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)

# Partición 80/20 por artículo (la primera de las 5 particiones agrupadas ≈ 20 %)
idx_train, idx_test = next(cv_grupos.split(X_txt, y, grupos))
X_train = [X_txt[i] for i in idx_train]; X_test = [X_txt[i] for i in idx_test]
y_train, y_test = y[idx_train], y[idx_test]
print(f"train: {len(idx_train)} párrafos · test: {len(idx_test)} párrafos "
      f"({len(set(grupos[idx_test]))} artículos que el modelo nunca ha visto)")

modelo = clone(busqueda.best_estimator_)
# La SVM no da probabilidades: si gana, se calibra para poder usar predict_proba
if not hasattr(modelo.named_steps["clf"], "predict_proba"):
    modelo.steps[-1] = ("clf", CalibratedClassifierCV(modelo.named_steps["clf"], cv=3))
t0 = time.time()
modelo.fit(X_train, y_train)
print(f"Entrenado en {time.time() - t0:.2f} s")

y_pred = modelo.predict(X_test)
print(f"\\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.3f}\\n")
print(classification_report(y_test, y_pred, digits=3))
""")
code("""
clases = list(CATEGORIES)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=clases), display_labels=clases).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Matriz de confusión (valores absolutos)")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=clases, normalize="true"),
                       display_labels=clases).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
axes[1].set_title("Normalizada por clase real (recall)")
plt.tight_layout(); plt.show()
""")
md("""
### 3.4 Interpretación: palabras más discriminantes

Los pesos del modelo lineal muestran qué lemas empujan hacia cada clase. Es una
comprobación de que el modelo ha aprendido vocabulario temático y no artefactos.
""")
code("""
modelo_interpretable = clone(busqueda.best_estimator_).fit(X_train, y_train)
top = top_features(modelo_interpretable, n=15)
display(pd.DataFrame(top)[clases])
""")

md("""
### 3.5 Medición del ruido de etiquetado con el clasificador

Segunda medida del ruido de la Fase 1: con **predicciones fuera de la partición**
(*out-of-fold*, cada párrafo lo predice un modelo que no ha visto su artículo),
se cuentan los párrafos en los que el modelo **contradice la etiqueta con mucha
seguridad** (probabilidad > 0.8 para otra clase). Son los candidatos a párrafos
genéricos o mal etiquetados.
""")
code("""
from sklearn.model_selection import cross_val_predict

modelo_oof = clone(modelo)
proba_oof = cross_val_predict(modelo_oof, X_txt, y, groups=grupos, cv=cv_grupos,
                              method="predict_proba", n_jobs=-1)
clases_oof = np.array(sorted(set(y)))
pred_oof = clases_oof[proba_oof.argmax(1)]
conf_oof = proba_oof.max(1)
df["pred_oof"], df["conf_oof"] = pred_oof, conf_oof
df["ruido_probable"] = (pred_oof != y) & (conf_oof > 0.8)

tabla_ruido = pd.DataFrame({
    "% contradice etiqueta (>0.8)": df.groupby("class")["ruido_probable"].mean() * 100,
    "% sin término de dominio": df.groupby("class")["sin_termino_de_dominio"].mean() * 100,
}).round(1).loc[clases]
display(tabla_ruido)
print(f"Estimación global de ruido (clasificador): {df['ruido_probable'].mean()*100:.1f} % de los párrafos")
print(f"Solo el {df.loc[df['ruido_probable'], 'sin_termino_de_dominio'].mean()*100:.0f} % de esos párrafos "
      "carece de términos de dominio: las dos medidas detectan cosas distintas (ver abajo).")
print("\\nEjemplos de párrafos 'ruidosos':")
for _, r in df[df["ruido_probable"]].sample(6, random_state=SEED).iterrows():
    print(f" · etiqueta={r['class']:<8} predicción={r['pred_oof']:<8} ({r['conf_oof']:.2f}) [{r['title']}]")
    print(f"   {r['text'][:170]}")
""")
md("""
La medida heurística (sin términos de dominio) es mucho más alta porque la
lista de raíces es corta: marca como "genérico" cualquier párrafo que use otro
vocabulario. El clasificador es más fino. Al revisar sus ejemplos se ven dos
tipos de "errores":

1. **Ruido real del etiquetado**: párrafos de historia, fechas o biografía que
   no hablan de la ciencia del artículo.
2. **Solapamiento real entre clases**: un párrafo de un artículo de química que
   habla de energía o de luz es, de verdad, de física. Aquí el "error" del
   modelo es razonable.

¿Eliminar estos párrafos? Se ha decidido **no** hacerlo en el conjunto de test
(sería evaluar el modelo con datos elegidos por el propio modelo), pero sí se
comprueba qué pasa si se entrenan sin ellos:
""")
code("""
mascara_train_limpio = ~df["ruido_probable"].values[idx_train]
modelo_limpio = clone(modelo).fit([X_train[i] for i in np.where(mascara_train_limpio)[0]],
                                  y_train[mascara_train_limpio])
print(f"F1 macro en test, entrenando con todo        : {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"F1 macro en test, entrenando sin párrafos ruidosos: "
      f"{f1_score(y_test, modelo_limpio.predict(X_test), average='macro'):.3f}")
""")

md("""
### 3.6 Modelo final con todos los datos

Para clasificar páginas web se reentrena el modelo con **todo** el dataset
(la evaluación ya se ha hecho con la partición de test).
""")
code("""
modelo_final = clone(modelo).fit(X_txt, y)


def clasificar_texto(texto):
    proba = modelo_final.predict_proba([" ".join(doc_to_features(nlp(texto), REP))])[0]
    return {str(c): round(float(p), 3) for c, p in zip(modelo_final.classes_, proba)}


for frase in ["La elektrono havas negativan elektran ŝargon kaj tre malgrandan mason.",
              "Natrio reakcias kun kloro kaj formas natrian kloridon, t.e. kuirsalon.",
              "La fotosintezo okazas en la kloroplastoj de la plantaj ĉeloj."]:
    print(clasificar_texto(frase), "←", frase)
""")

# ==========================================================================
# WEB
# ==========================================================================
md("""
## 4. Clasificación de páginas web

Se eligen artículos de Vikipedio **que no están en el dataset** (se comprueba
abajo) y se obtienen mediante *web scraping* de la **página HTML** con `requests`
+ `BeautifulSoup` (no con el API): se eliminan `script`, `style`, tablas,
navegación, referencias y cajas de información, y se extraen los párrafos `<p>`,
que se limpian con la misma función que el dataset.

Elegimos una página "fácil" por clase y una **ambigua** a propósito:
*Fotosintezo* (fotosíntesis) es un proceso biológico pero descrito con mucho
vocabulario químico y de energía.

**Robustez**: si una web falla (sin conexión, error HTTP, cambio de formato),
se usa una copia de respaldo publicada en el repositorio; si eso también falla,
esa página se omite con un aviso, pero el cuaderno sigue.
""")
code("""
PAGINAS = {
    "https://eo.wikipedia.org/wiki/Nigra_truo": "fiziko",     # agujero negro
    "https://eo.wikipedia.org/wiki/Acido": "kemio",           # ácido
    "https://eo.wikipedia.org/wiki/Mitozo": "biologio",       # mitosis
    "https://eo.wikipedia.org/wiki/Fotosintezo": "biologio",  # fotosíntesis (ambigua)
}
titulos_dataset = set(df["title"])
for url in PAGINAS:
    titulo = requests.utils.unquote(url.rsplit("/", 1)[1]).replace("_", " ")
    print(f"{titulo:<12} ¿está en el dataset? {'SÍ ⚠️' if titulo in titulos_dataset else 'no ✅'}")

respaldo = None


def obtener_parrafos(url):
    global respaldo
    html = fetch_html(url)
    if html:
        parrafos = html_to_paragraphs(html)
        if parrafos:
            return parrafos, "web en vivo"
    if respaldo is None:  # se descarga una sola vez
        try:
            respaldo = requests.get(URL_WEB_SNAPSHOT, timeout=20).json()
        except Exception as exc:
            print(f"  [aviso] tampoco hay copia de respaldo: {type(exc).__name__}")
            respaldo = {}
    return respaldo.get(url, []), "copia de respaldo"
""")
code("""
resultados = []
detalle = {}
for url, esperado in PAGINAS.items():
    parrafos, origen = obtener_parrafos(url)
    nombre = url.rsplit("/", 1)[1].replace("_", " ")
    if not parrafos:
        print(f"[omitida] {nombre}: no se pudo obtener texto")
        continue
    X_web = [" ".join(doc_to_features(d, REP)) for d in nlp.pipe(parrafos)]
    proba_par = modelo_final.predict_proba(X_web)
    proba_pag = proba_par.mean(axis=0)   # probabilidad de la página = media de sus párrafos
    detalle[nombre] = pd.DataFrame(proba_par, columns=modelo_final.classes_).assign(
        texto=[p[:80] + "…" for p in parrafos])
    fila = {"página": nombre, "origen": origen, "párrafos": len(parrafos), "esperada": esperado}
    fila.update({c: p for c, p in zip(modelo_final.classes_, proba_pag)})
    fila["predicción"] = modelo_final.classes_[proba_pag.argmax()]
    resultados.append(fila)

tabla_web = pd.DataFrame(resultados).set_index("página")
display(tabla_web.style.format({c: "{:.1%}" for c in clases})
        .background_gradient(subset=clases, cmap="Greens", axis=1))
""")
code("""
if len(tabla_web):
    ax = tabla_web[clases].plot.barh(stacked=True, figsize=(9, 0.6 + 0.6 * len(tabla_web)),
                                     color=[COLORES[c] for c in clases])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("probabilidad")
    ax.set_title("Probabilidad de cada categoría por página web")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=3, frameon=False)
    plt.show()
""")
md("""
**Probabilidad de la página = media de las probabilidades de sus párrafos.**
Se probó también a clasificar la página entera como un único documento, pero
Naive Bayes multiplica la evidencia de miles de palabras y la probabilidad se
satura a 0 % / 100 %, lo que no informa de la mezcla de temas. La media de
párrafos sí lo hace (y el modelo se entrenó precisamente con párrafos).

La clasificación **párrafo a párrafo** de la página ambigua muestra qué partes
tiran hacia cada categoría:
""")
code("""
for nombre in list(detalle)[-1:]:   # la página ambigua
    d = detalle[nombre]
    d["predicción"] = d[clases].idxmax(axis=1)
    print(f"{nombre}: reparto de párrafos →", d["predicción"].value_counts().to_dict())
    display(d[["texto"] + clases + ["predicción"]].head(10).style.format({c: "{:.2f}" for c in clases}))
""")

# ==========================================================================
# CONCLUSIONES
# ==========================================================================
md("""
## 5. Conclusiones

* **Dataset**: más de 1 500 párrafos por clase obtenidos automáticamente de
  Vikipedio, con un *script* reproducible y tolerante a fallos. El etiquetado
  heredado del artículo introduce ruido, que se ha medido de dos formas
  (vocabulario de dominio y desacuerdo del clasificador).
* **Modelo de NLP**: la regularidad del Esperanto permite un tokenizador,
  lematizador, etiquetador morfológico y gestor de stop-words basado en reglas,
  con un API como el de spaCy (`nlp(texto)` → `Doc` → `Token`), que acierta casi
  todos los tokens del *gold standard*.
* **Clasificación**: las representaciones que usan el modelo de NLP (lema o
  raíz) mejoran ligeramente a la palabra sin procesar y reducen el vocabulario
  casi a la mitad. Naive Bayes es el mejor modelo y consigue un F1 macro de
  ≈0.80 en validación cruzada por artículo y ≈0.85 en el test. La validación
  aleatoria por párrafos daría ≈0.89, una cifra inflada por la fuga de
  información entre párrafos del mismo artículo.
* **Web**: las tres páginas "claras" (*Nigra truo*, *Acido*, *Mitozo*) se
  clasifican correctamente con mucha seguridad. La página ambigua,
  *Fotosintezo*, sale como **química**: la mayoría de sus párrafos describen
  reacciones, moléculas y energía (es, en realidad, bioquímica). Es un resultado
  razonable, que además se explica por una decisión de la Fase 1: se excluyó la
  subcategoría *Biokemio* del dataset, así que el modelo asocia ese vocabulario
  a la química.

**Limitaciones y mejoras posibles**:

* El léxico cerrado no es exhaustivo y las reglas de participios generan algún
  falso positivo (*konstanta*), que se corrige con una lista de excepciones.
* El *gold standard* de la Fase 2 es pequeño (10 frases); un 100 % en él no
  significa que el modelo no se equivoque en textos reales.
* Algunos artículos muy largos pesan demasiado en su clase (p. ej. *masaĝ* entre
  las palabras más discriminantes de biología viene de un único artículo largo).
  Se podría limitar el número de párrafos por artículo.
* El dataset se podría limpiar más eliminando los párrafos genéricos detectados
  en 3.5, aunque en nuestras pruebas apenas cambia el resultado.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Cuaderno generado: {OUT.relative_to(ROOT)} ({len(cells)} celdas)")
