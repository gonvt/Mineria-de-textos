"""Tests de la Fase 1 (limpieza, segmentación y contrato de datos)."""

import pandas as pd

from eonlp.dataset import (EXCLUDED_SUBCATEGORIES, build_dataset, clean_paragraph,
                           load_dataset, save_dataset, split_paragraphs)

ARTICLE = """Fiziko estas la natura scienco kiu studas la materion, la energion, la spacon kaj la tempon kaj iliajn interagojn en la universo.

== Historio ==
La antikvaj grekoj jam studis la movadon de la korpoj kaj la naturon de la lumo; Aristotelo verkis libron pri fiziko [1].
Mallonga linio.

== Vidu ankaŭ ==
Ĉi tiu teksto ne devas aperi en la datumaro ĉar ĝi estas en la fina sekcio de la artikolo pri fiziko.
"""


def test_split_paragraphs_skips_headings_and_end_sections():
    ps = split_paragraphs(ARTICLE)
    assert len(ps) == 3
    assert not any("Vidu" in p or "fina sekcio" in p for p in ps)


def test_clean_paragraph():
    p = clean_paragraph("La antikvaj grekoj; jam studis [1] la movadon https://example.org de la "
                        "korpoj kaj la naturon de la lumo kaj de la steloj.")
    assert ";" not in p and "[1]" not in p and "http" not in p
    assert clean_paragraph("Tro mallonga.") is None
    assert clean_paragraph("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20") is None


def test_build_dataset_contract(tmp_path):
    raw = {
        "fiziko": [{"title": "Fiziko", "text": ARTICLE}, {"title": "Komuna", "text": ARTICLE}],
        "kemio": [{"title": "Komuna", "text": ARTICLE + "\nKemio estas la scienco pri la substancoj, "
                   "iliaj ecoj kaj la reakcioj per kiuj ili transformiĝas en aliajn substancojn."}],
    }
    df = build_dataset(raw)
    assert "Komuna" not in set(df["title"])        # artículo en dos clases -> descartado
    assert set(df["class"]) == {"fiziko"}
    path = tmp_path / "d.csv"
    save_dataset(df, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "text;class"
    back = load_dataset(path)
    assert list(back.columns) == ["text", "class"]
    assert len(back) == len(df)


def test_excluded_subcategories():
    assert EXCLUDED_SUBCATEGORIES.search("Kategorio:Fizikistoj")
    assert EXCLUDED_SUBCATEGORIES.search("Kategorio:Biokemio")
    assert not EXCLUDED_SUBCATEGORIES.search("Kategorio:Kemiaj elementoj")


def test_published_dataset_meets_contract():
    df = pd.read_csv("data/dataset.csv", sep=";", encoding="utf-8")
    assert list(df.columns) == ["text", "class"]
    assert not df["text"].str.contains(";").any()
    assert (df["class"].value_counts() >= 300).all()
    assert df["class"].nunique() >= 3
