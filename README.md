# Clasificación temática de textos en Esperanto

Práctica de programación de **Minería de Textos** (4º INF, 2026/2027).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gonvt/Mineria-de-textos/blob/main/notebooks/clasificacion_esperanto.ipynb)

Sistema completo que clasifica textos en Esperanto según el **deporte** del
que hablan:

| Clase | Fuente principal | Párrafos | Artículos |
|---|---|---:|---:|
| `teniso` (tenis) | [Kategorio:Teniso](https://eo.wikipedia.org/wiki/Kategorio:Teniso) + buscador | 966 | 490 |
| `golfo` (golf) | [Kategorio:Golfo (sporto)](https://eo.wikipedia.org/wiki/Kategorio:Golfo_(sporto)) + buscador | 218 | 135 |
| `rugbeo` (rugby) | [Kategorio:Rugbeo](https://eo.wikipedia.org/wiki/Kategorio:Rugbeo) + buscador | 328 | 121 |

Los tres deportes son **semánticamente cercanos** (jugadores, torneos,
campeones, pelota, olimpiadas...), así que el problema no es trivial.

> **¿Y el pádel?** La temática elegida era *pádel, golf y rugby*, pero la
> Vikipedio en Esperanto solo tiene **un artículo de pádel de cuatro frases** y
> no se encontró otra fuente en Esperanto, así que se sustituyó por el **tenis**,
> el deporte más parecido. El artículo de pádel se usa en la prueba web: el
> modelo lo clasifica como tenis.

## Las tres fases

1. **Dataset**: descarga automática de Vikipedio con su API. Se recorre la
   categoría de cada deporte y, como son pequeñas, se añaden los artículos del
   buscador que lo mencionan (de esos solo se guardan los párrafos que hablan
   del deporte). Después se divide en párrafos, se limpia y se mide el ruido de
   etiquetado. Resultado: `data/dataset.csv` (`text;class`).
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
| F1 macro, validación cruzada por artículo (SVM lineal + raíces) | 0.84 |
| Accuracy / F1 macro en test (25 % de los artículos, nunca vistos) | 0.88 / 0.87 |
| F1 macro CV sin las palabras *tenis-/golf-/rugbe-* | 0.66 |
| Exactitud del etiquetador POS en el *gold standard* | 100 % (73 tokens) |
| Tiempo de ejecución del cuaderno completo | ≈ 20 s (sin GPU) |

| Página web | Esperada | Predicha |
|---|---|---|
| [Noticia de Vikinovaĵoj: Coco Gauff gana el US Open](https://eo.wikinews.org/wiki/Usona_tenisisto_Coco_Gauff_gajnis_la_tenisturniron_US_Open) | teniso | teniso (94 %) |
| [Golfejo](https://eo.wikipedia.org/wiki/Golfejo) | golfo | golfo (99 %) |
| [Rugbea Monda Pokalo](https://eo.wikipedia.org/wiki/Rugbea_Monda_Pokalo) | rugbeo | rugbeo (98 %) |
| [Padelo (sporto)](https://eo.wikipedia.org/wiki/Padelo_(sporto)) | (pádel, sin clase) | teniso (100 %) |

**Limitaciones**: el golf tiene menos de los 300 párrafos recomendados (no hay
más texto en Esperanto), hay desbalanceo hacia el tenis y el modelo depende
bastante del propio nombre del deporte. Todo esto se mide y se discute en el
cuaderno.

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
│   ├── dataset_con_titulos.csv           el mismo dataset + título y fuente de cada párrafo
│   └── web_snapshot.json                 copia de respaldo de las 4 páginas web clasificadas
├── tests/                              ← 51 tests con pytest
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
(≈15 min). Si una de las dos fuentes falla, se usa la otra.

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
