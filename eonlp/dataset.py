"""Fase 1 - Construcción del dataset a partir de Vikipedio (Wikipedia en Esperanto).

Flujo:

1. ``collect_titles``: recorre una categoría de Wikipedia (y sus subcategorías
   hasta ``max_depth`` niveles, en anchura) y recoge títulos de artículos.
2. ``fetch_extract``: descarga el texto plano de cada artículo (API ``extracts``).
3. ``split_paragraphs`` + ``clean_paragraph``: divide en párrafos y limpia
   referencias, URLs, fórmulas, secciones finales ("Vidu ankaŭ", "Referencoj"...).
4. ``build_dataset``: une todo, elimina duplicados y artículos que aparecen en
   más de una clase, y devuelve un ``DataFrame`` con columnas ``text;class``.

Contrato de datos: CSV UTF-8, separador ``;``, cabecera ``text;class``,
un párrafo por fila, sin ``;`` en el texto y sin HTML/links/referencias.
"""

import re
import time

import pandas as pd

API_URL = "https://eo.wikipedia.org/w/api.php"
USER_AGENT = "MineriaDeTextos-Esperanto/1.0 (proyecto universitario; python-requests)"

# Categorías elegidas: tres ramas de las ciencias naturales (semánticamente cercanas)
CATEGORIES = {
    "fiziko": "Kategorio:Fiziko",
    "kemio": "Kategorio:Kemio",
    "biologio": "Kategorio:Biologio",
}

# Subcategorías que NO se recorren: biografías, listas, organizaciones,
# esbozos y temas mixtos (biofísica, bioquímica...) que mezclarían clases.
EXCLUDED_SUBCATEGORIES = re.compile(
    r"(istoj|ologoj|Listoj|Ĝermoj|Ŝablono|organizaĵoj|asocioj|premioj|fikcio|Magio|"
    r"Artikoloj sen|datumbazoj|socio|Biofiziko|Biokemio|Kvantuma kemio|Historio de|"
    r"Filozofio de|literaturo|kulturo|Metalurgio|industrio|inĝenierarto)",
    re.IGNORECASE,
)

# Secciones finales sin contenido temático
END_SECTIONS = re.compile(
    r"^(Vidu ankaŭ|Referencoj|Notoj|Eksteraj ligiloj|Ligiloj|Literaturo|Fontoj|"
    r"Bibliografio|Piednotoj|Rilataj artikoloj|Plia legado)\b",
    re.IGNORECASE,
)

MIN_WORDS = 15       # párrafos más cortos se descartan (títulos, pies de figura...)
MAX_WORDS = 300      # párrafos muy largos se recortan a esta longitud


# --------------------------------------------------------------------------
# Acceso al API de Wikipedia
# --------------------------------------------------------------------------
def make_session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def api_get(session, params, retries=4, timeout=30, pause=1.5):
    """Llamada GET al API con reintentos y espera exponencial.

    Devuelve el JSON o ``None`` si todos los intentos fallan (no lanza excepción:
    el cuaderno debe sobrevivir a fallos de red).
    """
    params = {"format": "json", "formatversion": "2", **params}
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001 - cualquier fallo de red se reintenta
            pass
        time.sleep(pause * (attempt + 1))
    return None


def category_members(session, category, member_type):
    """Lista los títulos de una categoría (``member_type`` = ``page`` o ``subcat``)."""
    titles, cont = [], {}
    while True:
        params = {
            "action": "query", "list": "categorymembers", "cmtitle": category,
            "cmtype": member_type, "cmnamespace": "0" if member_type == "page" else "14",
            "cmlimit": "500", **cont,
        }
        data = api_get(session, params)
        if not data:
            break
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" in data:
            cont = {"cmcontinue": data["continue"]["cmcontinue"]}
        else:
            break
    return titles


def collect_titles(session, root, max_articles=300, max_depth=2):
    """Recorrido en anchura (BFS) de la categoría ``root`` y sus subcategorías."""
    seen_cats = {root}
    queue = [(root, 0)]
    articles = []
    while queue and len(articles) < max_articles:
        cat, depth = queue.pop(0)
        for title in category_members(session, cat, "page"):
            if title.startswith("Listo"):
                continue
            if title not in articles:
                articles.append(title)
            if len(articles) >= max_articles:
                break
        if depth < max_depth:
            for sub in category_members(session, cat, "subcat"):
                if sub not in seen_cats and not EXCLUDED_SUBCATEGORIES.search(sub):
                    seen_cats.add(sub)
                    queue.append((sub, depth + 1))
    return articles


def fetch_extract(session, title):
    """Texto plano completo de un artículo (sin HTML gracias a ``explaintext``)."""
    data = api_get(session, {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "exsectionformat": "wiki", "redirects": "1", "titles": title,
    })
    try:
        return data["query"]["pages"][0].get("extract", "") or ""
    except (TypeError, KeyError, IndexError):
        return ""


def download_articles(categories=None, max_articles=300, max_depth=2, delay=0.1, verbose=True):
    """Descarga los artículos de cada clase: ``{clase: [{title, text}, ...]}``."""
    categories = categories or CATEGORIES
    session = make_session()
    raw = {}
    for label, cat in categories.items():
        titles = collect_titles(session, cat, max_articles, max_depth)
        raw[label] = []
        for i, title in enumerate(titles):
            raw[label].append({"title": title, "text": fetch_extract(session, title)})
            time.sleep(delay)
            if verbose and (i + 1) % 50 == 0:
                print(f"  {label}: {i + 1}/{len(titles)} artículos")
        if verbose:
            print(f"{label}: {len(titles)} artículos descargados")
    return raw


# --------------------------------------------------------------------------
# Limpieza y segmentación en párrafos
# --------------------------------------------------------------------------
_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REF = re.compile(r"\[\s*(\d+|[a-z]|noto \d+|citaĵo bezonata)\s*\]", re.IGNORECASE)
_PARENS_EMPTY = re.compile(r"\(\s*[,;:]?\s*\)")
_MATH = re.compile(r"\{\\displaystyle[^}]*\}|\\[a-zA-Z]+")
_SPACES = re.compile(r"\s+")


def split_paragraphs(text):
    """Divide un extracto en párrafos, ignorando títulos y secciones finales."""
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            if END_SECTIONS.match(m.group(1)):
                break            # a partir de aquí: bibliografía, enlaces...
            continue             # título de sección: no es un párrafo
        paragraphs.append(line)
    return paragraphs


def alpha_ratio(text):
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text.replace(" ", "")))


def clean_paragraph(p, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Limpia un párrafo y devuelve ``None`` si no es válido para el dataset."""
    p = _URL.sub(" ", p)
    p = _REF.sub(" ", p)
    p = _MATH.sub(" ", p)
    p = p.replace(";", ",")          # contrato de datos: sin ';'
    p = p.replace('"', "'")
    p = _PARENS_EMPTY.sub(" ", p)
    p = _SPACES.sub(" ", p).strip()
    words = p.split()
    if len(words) < min_words:
        return None
    if alpha_ratio(p) < 0.75:        # tablas, fórmulas, listas de cifras
        return None
    if len(words) > max_words:
        p = " ".join(words[:max_words])
    return p


def build_dataset(raw, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Convierte los artículos descargados en un ``DataFrame`` ``text;class``.

    Decisiones:

    * Un artículo que aparece en varias clases se descarta (etiqueta ambigua).
    * Párrafos duplicados se eliminan (plantillas repetidas en varios artículos).
    * Se guarda también el título de origen (columna ``title``) para poder
      hacer particiones por artículo y analizar el ruido; el CSV final del
      contrato solo contiene ``text;class``.
    """
    owners = {}
    for label, arts in raw.items():
        for a in arts:
            owners.setdefault(a["title"], set()).add(label)
    ambiguous = {t for t, labels in owners.items() if len(labels) > 1}

    rows = []
    for label, arts in raw.items():
        for a in arts:
            if a["title"] in ambiguous or not a.get("text"):
                continue
            for p in split_paragraphs(a["text"]):
                c = clean_paragraph(p, min_words, max_words)
                if c:
                    rows.append({"text": c, "class": label, "title": a["title"]})
    df = pd.DataFrame(rows, columns=["text", "class", "title"])
    df = df.drop_duplicates(subset="text", keep=False).reset_index(drop=True)
    return df


def save_dataset(df, path):
    """Guarda el dataset cumpliendo el contrato de datos."""
    assert not df["text"].str.contains(";").any()
    df[["text", "class"]].to_csv(path, sep=";", index=False, encoding="utf-8")


def load_dataset(path_or_url):
    """Carga un CSV con el contrato ``text;class``."""
    return pd.read_csv(path_or_url, sep=";", encoding="utf-8")
