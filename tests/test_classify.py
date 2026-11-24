"""Tests de la Fase 3 (vectorización, clasificador y scraping)."""

from eonlp import load
from eonlp.classify import build_pipeline, candidate_models, doc_to_features, html_to_paragraphs, preprocess

TRAIN = {
    "teniso": ["La tenisisto venkis en la finalo de Wimbledon post kvin setoj kontraŭ la ĉampiono.",
               "Ŝi servis tre forte kaj gajnis la tenisan turniron sur la herba tenisejo.",
               "La rakedo kaj la pilko de teniso estas pli malpezaj ol antaŭ cent jaroj."],
    "golfo": ["La golfludisto bezonis tri batojn por meti la pilkon en la truon.",
              "La golfejo havas dek ok truojn kaj la golfbastonoj estas en la ĉaro.",
              "Li gajnis la golfturniron Masters post kvar tagoj de ludado."],
    "rugbeo": ["La rugbea teamo de Nov-Zelando gajnis la mondan pokalon.",
               "La rugbeistoj portas la ovalan pilkon trans la linion por fari provon.",
               "En la skramo ok rugbeistoj de ĉiu teamo puŝas kontraŭ la aliaj."],
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
    pred = model.predict(preprocess(["La rugbeistoj de la teamo portis la pilkon."], nlp))
    assert pred[0] == "rugbeo"
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
