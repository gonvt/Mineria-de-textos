# Clasificación temática de textos en Esperanto

Práctica de programación de **Minería de Textos** (4º INF, 2026/2027).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gonvt/Mineria-de-textos/blob/main/notebooks/clasificacion_esperanto.ipynb)

Sistema completo que clasifica textos en Esperanto según su temática científica:

| Clase | Fuente | Párrafos | Artículos |
|---|---|---:|---:|
| `fiziko` (física) | [Kategorio:Fiziko](https://eo.wikipedia.org/wiki/Kategorio:Fiziko) | 2 293 | 270 |
| `kemio` (química) | [Kategorio:Kemio](https://eo.wikipedia.org/wiki/Kategorio:Kemio) | 1 646 | 262 |
| `biologio` (biología) | [Kategorio:Biologio](https://eo.wikipedia.org/wiki/Kategorio:Biologio) | 1 579 | 283 |

Las tres categorías son ramas de las ciencias naturales, **semánticamente
cercanas**, para que el problema no sea trivial.

## Las tres fases

1. **Dataset**: descarga automática de artículos de Vikipedio con su API
   (recorrido de categorías y subcategorías), división en párrafos, limpieza y
   medición del ruido de etiquetado. Resultado: `data/dataset.csv` (`text;class`).
2. **NLP**: modelo propio para Esperanto, basado en reglas y **sin librerías de
   NLP**, con un API como el de spaCy: tokenizador, lematizador, etiquetador
   gramatical y morfológico (UPOS + rasgos), stop-words, objetos `Doc`/`Token`,
   segmentación en oraciones y descomposición en morfemas.
3. **Clasificación**: TF-IDF sobre lemas/raíces + Naive Bayes, regresión
   logística y SVM, con validación cruzada **agrupada por artículo**.
   Clasificación de páginas web reales con *web scraping*.

### Resultados principales

| | Valor |
|---|---|
| F1 macro, validación cruzada por artículo (5 particiones) | 0.81 |
| Accuracy / F1 macro en test (20 % de los artículos, nunca vistos) | 0.85 / 0.85 |
| Exactitud del etiquetador POS en el *gold standard* | 100 % (73 tokens) |
| Tiempo de ejecución del cuaderno completo | ≈ 20 s (sin GPU) |

| Página web | Esperada | Predicha |
|---|---|---|
| [Nigra truo](https://eo.wikipedia.org/wiki/Nigra_truo) | fiziko | fiziko (88 %) |
| [Acido](https://eo.wikipedia.org/wiki/Acido) | kemio | kemio (83 %) |
| [Mitozo](https://eo.wikipedia.org/wiki/Mitozo) | biologio | biologio (80 %) |
| [Fotosintezo](https://eo.wikipedia.org/wiki/Fotosintezo) | biologio (ambigua) | kemio (75 %) |

## Estructura del repositorio

```
Mineria-de-textos/
├── notebooks/
│   └── clasificacion_esperanto.ipynb   ← LA ENTREGA: cuaderno autocontenido (Colab)
├── eonlp/                              ← librería de Python instalable (extra)
│   ├── __init__.py                       API pública: load(), Doc, Token...
│   ├── lexicon.py                        léxico cerrado: pronombres, correlativos, preposiciones, stop-words
│   ├── nlp.py                            Fase 2: modelo de NLP (tokenizador, tagger, lematizador...)
│   ├── dataset.py                        Fase 1: descarga de Vikipedio, limpieza, contrato de datos
│   └── classify.py                       Fase 3: vectorización, modelos y web scraping
├── data/
│   ├── dataset.csv                       dataset final con el contrato de datos (text;class)
│   ├── dataset_con_titulos.csv           el mismo dataset + título del artículo (para agrupar)
│   └── web_snapshot.json                 copia de respaldo de las páginas web clasificadas
├── tests/                              ← 50 tests con pytest
│   ├── test_nlp.py                       tokenizador, lemas, POS, morfología, stop-words
│   ├── test_dataset.py                   limpieza, párrafos y contrato de datos
│   └── test_classify.py                  vectorización, entrenamiento y scraping
├── scripts/
│   └── build_notebook.py                 genera el cuaderno a partir del código de eonlp/
├── docs/
│   └── GUIA.md                           explicación detallada de cada parte (para la defensa)
├── pyproject.toml                        configuración del paquete (pip install)
└── requirements.txt                      dependencias
```

## Cómo usarlo

### En Google Colab (recomendado)

Pulsa el botón *Abrir en Colab* y ejecuta todo (*Entorno de ejecución →
Ejecutar todas*). No hace falta instalar nada.

El parámetro `GENERAR_DATASET` de la primera celda decide si se usa el dataset
ya publicado en `data/` (rápido) o si se descarga todo de nuevo desde Vikipedio
(≈10–15 min). Si una de las dos fuentes falla, se usa la otra.

### En local

```bash
git clone https://github.com/gonvt/Mineria-de-textos.git
cd Mineria-de-textos
pip install -e ".[dev]"
pytest                                   # ejecuta los tests
jupyter notebook notebooks/clasificacion_esperanto.ipynb
```

### Como librería

```bash
pip install git+https://github.com/gonvt/Mineria-de-textos.git
```

```python
import eonlp

nlp = eonlp.load()
doc = nlp("La junaj sciencistoj malkovris novajn elementojn.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)
# La la DET Definite=Def|PronType=Art True
# junaj juna ADJ Case=Nom|Number=Plur False
# sciencistoj sciencisto NOUN Case=Nom|Number=Plur False
# malkovris malkovri VERB Mood=Ind|Tense=Past|VerbForm=Fin False
# ...
```

### Regenerar el cuaderno tras cambiar la librería

El código del cuaderno se copia de `eonlp/`, así que la librería es la única
fuente del código:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/clasificacion_esperanto.ipynb
```

## Referencias

* Enunciado de la práctica y repositorio guía:
  [jparisu/nlp-esperantilo](https://github.com/jparisu/nlp-esperantilo).
* API de MediaWiki: <https://www.mediawiki.org/wiki/API:Main_page>.
* Etiquetas UPOS y rasgos morfológicos: [Universal Dependencies](https://universaldependencies.org/u/pos/).
