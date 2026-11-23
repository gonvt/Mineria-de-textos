"""Tests de la Fase 3 (vectorización, clasificador y scraping)."""

from eonlp import load
from eonlp.classify import build_pipeline, candidate_models, doc_to_features, html_to_paragraphs, preprocess

TRAIN = {
    "fiziko": ["La forto akcelas la mason laŭ la leĝo de Newton kaj la energio konserviĝas.",
               "La lumo estas elektromagneta ondo kun frekvenco kaj ondolongo.",
               "La gravito altiras la korpojn kaj la rapido de la falo kreskas."],
    "kemio": ["La acido reakcias kun la bazo kaj formas salon kaj akvon.",
              "La atomoj de oksigeno kaj hidrogeno formas molekulon de akvo.",
              "La kemia reakcio liberigas varmon kaj la elementoj kombiniĝas."],
    "biologio": ["La ĉeloj de la plantoj enhavas kloroplastojn kaj kromosomojn.",
                 "La bestoj kaj plantoj evoluis dum milionoj da jaroj per natura selektado.",
                 "La genoj de la organismo troviĝas en la DNA de ĉiu ĉelo."],
}


def test_doc_to_features_removes_stopwords():
    nlp = load()
    feats = doc_to_features(nlp("La atomoj estas tre malgrandaj kaj ili moviĝas."), "lemma")
    assert feats == ["atomo", "moviĝi"]  # "malgranda" está en la lista de stop-words
    assert doc_to_features(nlp("fizikaj fizike"), "root") == ["fizik", "fizik"]


def test_pipeline_trains_and_predicts():
    nlp = load()
    texts = [t for ts in TRAIN.values() for t in ts]
    labels = [c for c, ts in TRAIN.items() for _ in ts]
    X = preprocess(texts, nlp)
    model = build_pipeline(candidate_models()["Naive Bayes"], min_df=1, max_df=1.0).fit(X, labels)
    pred = model.predict(preprocess(["La ĉeloj kaj la genoj de la plantoj."], nlp))
    assert pred[0] == "biologio"
    assert abs(model.predict_proba(X).sum(axis=1) - 1).max() < 1e-9


def test_html_to_paragraphs():
    html = """<html><body><nav>menuo menuo</nav><div id="mw-content-text">
    <p>La fotosintezo estas procezo en la plantoj<sup class="reference">[1]</sup>, per kiu la lumenergio
    transformiĝas en kemian energion; ĝi okazas en la kloroplastoj.</p>
    <table><tr><td>tabelo kun multaj vortoj kiuj ne devas aperi en la rezulto ĉar ĝi estas tabelo</td></tr></table>
    <p>Mallonga.</p><script>var x = 1;</script></div></body></html>"""
    ps = html_to_paragraphs(html)
    assert len(ps) == 1
    assert "[1]" not in ps[0] and ";" not in ps[0] and "plantoj," in ps[0]
