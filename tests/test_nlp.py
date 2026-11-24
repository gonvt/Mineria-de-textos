"""Tests del modelo de NLP para Esperanto (Fase 2)."""

import pytest

from eonlp import Doc, Token, load, normalize_text, split_morphemes


@pytest.fixture(scope="module")
def nlp():
    return load()


def analyze(nlp, text):
    return [(t.text, t.lemma_, t.pos_) for t in nlp(text)]


# ---------------------------------------------------------------- API spaCy
def test_api_like_spacy(nlp):
    doc = nlp("La katoj manĝis fiŝon.")
    assert isinstance(doc, Doc)
    assert isinstance(doc[0], Token)
    assert len(doc) == 5
    assert doc.text == "La katoj manĝis fiŝon."
    assert [t.i for t in doc] == list(range(5))
    assert nlp.pipe_names == ["normalizer", "tokenizer", "tagger", "lemmatizer", "stopwords"]
    assert "".join(t.text_with_ws for t in doc) == doc.text


def test_pipe(nlp):
    docs = list(nlp.pipe(["Mi legas.", "Vi skribas."]))
    assert [d[1].lemma_ for d in docs] == ["legi", "skribi"]


def test_sentences(nlp):
    doc = nlp("Mi venas. Ĉu vi venas? Jes!")
    assert [s.text for s in doc.sents] == ["Mi venas.", "Ĉu vi venas?", "Jes!"]


# ------------------------------------------------------------ tokenizador
def test_tokenizer_punctuation_numbers_and_elision(nlp):
    assert [t.text for t in nlp("En 1887, Zamenhof publikigis la libron.")] == [
        "En", "1887", ",", "Zamenhof", "publikigis", "la", "libron", "."]
    assert [t.text for t in nlp("de l' mondo")] == ["de", "l'", "mondo"]
    assert [t.text for t in nlp("dek-du homoj")][0] == "dek-du"


def test_x_system_normalization():
    assert normalize_text("Cxu vi sxatas gxin?") == "Ĉu vi ŝatas ĝin?"
    assert normalize_text("auxto") == "auxto".replace("ux", "ŭ")


# ----------------------------------------------------- etiquetado y lemas
@pytest.mark.parametrize("word,lemma,pos", [
    ("katoj", "kato", "NOUN"),
    ("fiŝon", "fiŝo", "NOUN"),
    ("domojn", "domo", "NOUN"),
    ("bela", "bela", "ADJ"),
    ("interesajn", "interesa", "ADJ"),
    ("rapide", "rapide", "ADV"),
    ("manĝas", "manĝi", "VERB"),
    ("manĝis", "manĝi", "VERB"),
    ("manĝos", "manĝi", "VERB"),
    ("manĝus", "manĝi", "VERB"),
    ("manĝu", "manĝi", "VERB"),
    ("manĝi", "manĝi", "VERB"),
])
def test_open_class_words(nlp, word, lemma, pos):
    tok = nlp("mi " + word)[1]
    assert (tok.lemma_, tok.pos_) == (lemma, pos)


@pytest.mark.parametrize("word,lemma,pos", [
    ("la", "la", "DET"), ("kaj", "kaj", "CCONJ"), ("ke", "ke", "SCONJ"),
    ("en", "en", "ADP"), ("mi", "mi", "PRON"), ("min", "mi", "PRON"),
    ("mia", "mia", "DET"), ("miajn", "mia", "DET"), ("kiu", "kiu", "DET"),
    ("tiujn", "tiu", "DET"), ("nenio", "nenio", "PRON"), ("kiam", "kiam", "ADV"),
    ("ankaŭ", "ankaŭ", "ADV"), ("tri", "tri", "NUM"), ("dudek", "dudek", "NUM"),
])
def test_closed_class_words(nlp, word, lemma, pos):
    tok = nlp("x " + word)[1]
    assert (tok.lemma_, tok.pos_) == (lemma, pos)


def test_morphology(nlp):
    tok = nlp("mi vidas belajn domojn")[3]
    assert tok.morph.get("Number") == "Plur"
    assert tok.morph.get("Case") == "Acc"
    assert str(tok.morph) == "Case=Acc|Number=Plur"
    assert nlp("li iris")[1].morph.get("Tense") == "Past"
    assert nlp("li irus")[1].morph.get("Mood") == "Cnd"


def test_participles(nlp):
    tok = nlp("la skribita letero")[1]
    assert tok.lemma_ == "skribi"
    assert tok.morph.get("VerbForm") == "Part"
    assert tok.morph.get("Voice") == "Pass"
    assert nlp("la konstanta valoro")[1].lemma_ == "konstanta"  # excepción


def test_proper_nouns(nlp):
    doc = nlp("Ŝi loĝas en Parizo kun Marie Curie.")
    assert [t.pos_ for t in doc][3:6] == ["PROPN", "ADP", "PROPN"]
    assert doc[0].pos_ == "PRON"           # mayúscula a inicio de oración no es PROPN
    assert nlp("la leĝo de Newton")[3].pos_ == "PROPN"


def test_ordinals(nlp):
    tok = nlp("la dua leĝo")[1]
    assert tok.pos_ == "ADJ" and tok.morph.get("NumType") == "Ord"


def test_spelling_variants(nlp):
    assert nlp("la ĥemio")[1].lemma_ == "kemio"
    assert nlp("bioĥemia procezo")[0].lemma_ == "biokemia"


# --------------------------------------------------------------- stop-words
def test_stop_words(nlp):
    doc = nlp("La atomo estas tre malgranda kaj ĝi ne estis videbla.")
    stops = {t.text for t in doc if t.is_stop}
    assert {"La", "estas", "tre", "kaj", "ĝi", "ne", "estis"} <= stops
    assert "atomo" not in stops and "videbla" not in stops


def test_add_stop_words():
    nlp = load()
    nlp.add_stop_words(["atomo"])
    assert nlp("atomo")[0].is_stop


# ----------------------------------------------------------------- morfemas
@pytest.mark.parametrize("stem,expected", [
    ("malsanulej", ["mal", "san", "ul", "ej"]),
    ("lernej", ["lern", "ej"]),
    ("ide", ["ide"]),
])
def test_split_morphemes(stem, expected):
    assert split_morphemes(stem) == expected
