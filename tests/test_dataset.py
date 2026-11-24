"""Tests de la Fase 1 (limpieza, segmentación y contrato de datos)."""

import re

import pandas as pd

from eonlp.dataset import (EXCLUDED_SUBCATEGORIES, TOPICS, build_dataset, clean_paragraph,
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


TENNIS = ("La teniso estas rakeda sporto kiun ludas du aŭ kvar ludantoj sur tenisejo, "
          "kaj la celo estas trafi la pilkon trans la reton.")
OLYMPICS = ("La Somera Olimpiko okazis en la urbo kun multaj sportoj kaj miloj da sportistoj "
            "el pli ol cent landoj de la mondo.\n"
            "En la rugbea turniro la teamo de Fiĝioj gajnis la oran medalon post bona finalo "
            "kontraŭ la teamo de Britio.\n"
            "La tenisa kaj rugbea turniroj okazis en la sama semajno en du malsamaj stadionoj "
            "de la olimpika parko.")


def test_build_dataset_rules(tmp_path):
    raw = {
        "teniso": [{"title": "Teniso", "source": "kategorio", "text": TENNIS + "\n" + ARTICLE}],
        "rugbeo": [{"title": "Olimpiko", "source": "sercho", "text": OLYMPICS},
                   {"title": "Golfejo", "source": "kategorio", "text": TENNIS}],
    }
    df = build_dataset(raw, exclude_titles={"Golfejo"})
    # categoría: se guardan todos los párrafos (también el genérico de ARTICLE)
    assert (df["title"] == "Teniso").sum() == 3
    # buscador: solo el párrafo que menciona el rugby y ningún otro deporte
    olimp = df[df["title"] == "Olimpiko"]
    assert len(olimp) == 1 and "Fiĝioj" in olimp["text"].iloc[0]
    assert "Golfejo" not in set(df["title"])             # página reservada
    path = tmp_path / "d.csv"
    save_dataset(df, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "text;class"
    back = load_dataset(path)
    assert list(back.columns) == ["text", "class"]
    assert len(back) == len(df)


def test_golf_keywords_ignore_geographic_gulf():
    rx = re.compile(TOPICS["golfo"]["keywords"], re.IGNORECASE)
    assert rx.search("Li estas fama usona golfludisto.")
    assert rx.search("La golfejo havas dek ok truojn.")
    assert not rx.search("La ŝipo eniris la Golfon de Meksiko.")


def test_excluded_subcategories():
    assert EXCLUDED_SUBCATEGORIES.search("Kategorio:Usona piedpilko")
    assert not EXCLUDED_SUBCATEGORIES.search("Kategorio:Tenisistoj")


def test_published_dataset_meets_contract():
    df = pd.read_csv("data/dataset.csv", sep=";", encoding="utf-8")
    assert list(df.columns) == ["text", "class"]
    assert not df["text"].str.contains(";").any()
    assert set(df["class"]) == set(TOPICS)
    assert not df["text"].duplicated().any()
