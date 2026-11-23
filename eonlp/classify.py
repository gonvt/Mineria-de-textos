"""Fase 3 - Vectorización, clasificación y clasificación de páginas web.

* ``preprocess``: usa el modelo de NLP propio para convertir cada párrafo en
  una secuencia de lemas (o raíces) de palabras con contenido.
* ``build_pipeline``: TF-IDF + clasificador lineal de scikit-learn.
* ``fetch_web_paragraphs``: *web scraping* con ``requests`` + ``BeautifulSoup``.
* ``classify_paragraphs``: probabilidad de cada categoría para un texto web.
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .dataset import clean_paragraph

CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}
SEED = 42


# --------------------------------------------------------------------------
# Preprocesamiento con el modelo de NLP propio
# --------------------------------------------------------------------------
def doc_to_features(doc, mode="lemma", min_len=3):
    """Extrae las "palabras" que representan un documento.

    ``mode``:
      * ``"lemma"``  - lema (``fizikaj`` -> ``fizika``; ``atomoj`` -> ``atomo``)
      * ``"root"``   - raíz sin terminación (``fizika``/``fiziko`` -> ``fizik``),
        aprovecha la morfología aglutinante del Esperanto.
      * ``"text"``   - palabra en minúsculas sin lematizar (línea base).
    Se descartan stop-words, signos de puntuación, números y palabras cortas.
    """
    feats = []
    for t in doc:
        if t.is_stop or t.pos_ not in CONTENT_POS or not t.is_alpha:
            continue
        value = {"lemma": t.lemma_, "root": t.root_, "text": t.lower_}[mode].lower()
        if len(value) >= min_len:
            feats.append(value)
    return feats


def preprocess(texts, nlp, mode="lemma"):
    """Devuelve una lista de cadenas de *features* separadas por espacios."""
    return [" ".join(doc_to_features(doc, mode)) for doc in nlp.pipe(texts)]


def preprocess_all_modes(texts, nlp, modes=("text", "lemma", "root")):
    """Igual que ``preprocess`` pero obtiene varias representaciones con una
    sola pasada del modelo de NLP (ahorra tiempo de ejecución)."""
    out = {m: [] for m in modes}
    for doc in nlp.pipe(texts):
        for m in modes:
            out[m].append(" ".join(doc_to_features(doc, m)))
    return out


def make_vectorizer(ngram_range=(1, 1), min_df=2, max_df=0.9):
    """TF-IDF sobre los lemas ya preprocesados (separados por espacios)."""
    return TfidfVectorizer(
        token_pattern=r"\S+", lowercase=False, ngram_range=ngram_range,
        min_df=min_df, max_df=max_df, sublinear_tf=True,
    )


def candidate_models():
    """Clasificadores ligeros (entrenan en segundos sin GPU)."""
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Regresión logística": LogisticRegression(C=10, max_iter=2000, random_state=SEED),
        "SVM lineal": LinearSVC(C=1.0, random_state=SEED),
    }


def build_pipeline(model, **vec_kwargs):
    return Pipeline([("tfidf", make_vectorizer(**vec_kwargs)), ("clf", model)])


def top_features(pipeline, n=15):
    """Palabras con más peso para cada clase (modelos lineales)."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    if weights.shape[0] == 1:  # caso binario
        weights = np.vstack([-weights[0], weights[0]])
    return {c: list(names[np.argsort(w)[::-1][:n]]) for c, w in zip(clf.classes_, weights)}


# --------------------------------------------------------------------------
# Web scraping
# --------------------------------------------------------------------------
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "table",
               "figure", "sup", "noscript", "form"]


def html_to_paragraphs(html, min_words=10):
    """Extrae los párrafos ``<p>`` limpios del HTML de una página."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    # En Wikipedia, eliminar cajas de navegación y referencias
    for sel in [".reference", ".mw-editsection", ".navbox", ".infobox", ".reflist", ".catlinks"]:
        for el in soup.select(sel):
            el.decompose()
    root = soup.select_one("#mw-content-text") or soup.find("article") or soup.find("main") or soup
    paragraphs = []
    for p in root.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text()).strip()
        text = re.sub(r"\s+([,.:!?)])", r"\1", text).replace("( ", "(")
        text = clean_paragraph(text, min_words=min_words)
        if text:
            paragraphs.append(text)
    return paragraphs


def fetch_html(url, timeout=20):
    """Descarga una página; devuelve ``None`` si falla (sin lanzar excepción)."""
    import requests

    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "MineriaDeTextos-Esperanto/1.0 (proyecto universitario)"})
        r.raise_for_status()
        r.encoding = r.apparent_encoding if not r.encoding else r.encoding
        return r.text
    except Exception as exc:  # noqa: BLE001
        print(f"  [aviso] no se pudo descargar {url}: {type(exc).__name__}")
        return None


def classify_paragraphs(pipeline, paragraphs, nlp, mode="lemma"):
    """Probabilidades por párrafo y para la página completa.

    La página completa se clasifica concatenando todos sus párrafos (un único
    documento) y, como comprobación, también se da la media de los párrafos.
    """
    X = preprocess(paragraphs, nlp, mode)
    proba = pipeline.predict_proba(X)
    per_par = pd.DataFrame(proba, columns=pipeline.classes_)
    per_par.insert(0, "texto", [p[:90] + ("…" if len(p) > 90 else "") for p in paragraphs])
    page = pipeline.predict_proba([" ".join(X)])[0]
    summary = pd.DataFrame({
        "página completa": page,
        "media de párrafos": proba.mean(axis=0),
    }, index=pipeline.classes_)
    return summary, per_par
