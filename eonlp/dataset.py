"""Fase 1 - Construcción del dataset a partir de Vikipedio (Wikipedia en Esperanto).

Flujo:

1. ``collect_titles``: recorre la categoría del deporte en Vikipedio (y sus
   subcategorías hasta ``max_depth`` niveles, en anchura) y recoge artículos.
2. ``search_titles``: como las categorías deportivas de Vikipedio son pequeñas,
   se añaden los artículos que **mencionan** el deporte (buscador del API).
3. ``fetch_extract``: descarga el texto plano de cada artículo (API ``extracts``).
4. ``split_paragraphs`` + ``clean_paragraph``: divide en párrafos y limpia
   referencias, URLs, fórmulas, secciones finales ("Vidu ankaŭ", "Referencoj"...).
5. ``build_dataset``: de los artículos de la categoría se guardan todos los
   párrafos; de los encontrados con el buscador, solo los párrafos que hablan
   del deporte (contienen una de sus palabras clave). Se eliminan duplicados y
   párrafos que mencionan dos deportes. Devuelve un ``DataFrame`` ``text;class``.

Contrato de datos: CSV UTF-8, separador ``;``, cabecera ``text;class``,
un párrafo por fila, sin ``;`` en el texto y sin HTML/links/referencias.
"""

import re
import time

import pandas as pd

API_URL = "https://eo.wikipedia.org/w/api.php"
USER_AGENT = "MineriaDeTextos-Esperanto/1.0 (proyecto universitario; python-requests)"

# Temática: DEPORTES. Tres deportes (semánticamente cercanos: todos tienen
# jugadores, torneos, campeones, pelota, puntos...).
#   category: categoría de Vikipedio que se recorre
#   search:   búsquedas para encontrar más artículos que mencionan el deporte
#   keywords: expresión regular que identifica un párrafo sobre ese deporte.
#             Ojo con "golfo": en Esperanto también es un golfo geográfico
#             ("Golfo de Meksiko"), por eso solo se aceptan derivados claros.
TOPICS = {
    "teniso": {
        "category": "Kategorio:Teniso",
        "search": ["teniso", "tenisisto", "tenisejo"],
        "keywords": r"tenis",
    },
    "golfo": {
        "category": "Kategorio:Golfo (sporto)",
        "search": ["golfludo", "golfludisto", "golfludanto", "golfejo", "golfisto", "golfklubo"],
        "keywords": r"golf(?:lud|ej|ist|bast|pilk|klub|turnir|ĉar|truo)|golfo \(sporto\)|lud\w* golfon",
    },
    "rugbeo": {
        "category": "Kategorio:Rugbeo",
        "search": ["rugbeo", "rugbea", "rugbeisto", "rugbeteamo"],
        "keywords": r"rugb",          # rugbeo, rugbea, rugbeisto... y "Rugby" (nombres propios)
    },
}
CATEGORIES = {label: cfg["category"] for label, cfg in TOPICS.items()}

# Páginas reservadas para la prueba de clasificación web (Fase 3): se excluyen
# del dataset para que el modelo no las haya visto al entrenar.
HOLDOUT_TITLES = {"Golfejo", "Rugbea Monda Pokalo", "Padelo (sporto)", "Somera Olimpiko 2016"}

# Subcategorías que NO se recorren: el fútbol americano y canadiense cuelga de
# "Rugbeo" en Vikipedio pero es otro deporte; videojuegos, esbozos y plantillas.
EXCLUDED_SUBCATEGORIES = re.compile(
    r"(Usona piedpilko|NFL|Usonpiedpilk|Kanada|videoludoj|Ĝermoj|Ŝablono)",
    re.IGNORECASE,
)

# Secciones finales sin contenido temático
END_SECTIONS = re.compile(
    r"^(Vidu ankaŭ|Referencoj|Notoj|Eksteraj ligiloj|Ligiloj|Literaturo|Fontoj|"
    r"Bibliografio|Piednotoj|Rilataj artikoloj|Plia legado)\b",
    re.IGNORECASE,
)

MIN_WORDS = 10       # párrafos más cortos se descartan (títulos, pies de figura...)
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


def search_titles(session, query, max_results=500):
    """Títulos de artículos cuyo texto contiene ``query`` (``list=search``)."""
    titles, offset = [], 0
    while len(titles) < max_results:
        data = api_get(session, {"action": "query", "list": "search", "srsearch": query,
                                 "srlimit": "500", "sroffset": str(offset), "srnamespace": "0"})
        if not data:
            break
        titles += [r["title"] for r in data["query"]["search"]]
        if "continue" in data:
            offset = data["continue"]["sroffset"]
        else:
            break
    return titles[:max_results]


def collect_titles(session, root, max_articles=400, max_depth=3):
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


def download_articles(topics=None, max_articles=400, max_depth=3, max_search=500,
                      delay=0.1, verbose=True):
    """Descarga los artículos de cada clase.

    Devuelve ``{clase: [{title, source, text}, ...]}`` donde ``source`` es
    ``"kategorio"`` (artículo de la categoría del deporte) o ``"sercho"``
    (artículo encontrado con el buscador).
    """
    topics = topics or TOPICS
    session = make_session()
    raw = {}
    for label, cfg in topics.items():
        cat_titles = collect_titles(session, cfg["category"], max_articles, max_depth)
        found = []
        for q in cfg.get("search", []):
            for t in search_titles(session, q, max_search):
                if t not in cat_titles and t not in found and not t.startswith("Listo"):
                    found.append(t)
        jobs = [(t, "kategorio") for t in cat_titles] + [(t, "sercho") for t in found]
        raw[label] = []
        for i, (title, source) in enumerate(jobs):
            raw[label].append({"title": title, "source": source, "text": fetch_extract(session, title)})
            time.sleep(delay)
            if verbose and (i + 1) % 100 == 0:
                print(f"  {label}: {i + 1}/{len(jobs)} artículos")
        if verbose:
            print(f"{label}: {len(cat_titles)} artículos de la categoría + {len(found)} del buscador")
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


def build_dataset(raw, topics=None, min_words=MIN_WORDS, max_words=MAX_WORDS,
                  exclude_titles=HOLDOUT_TITLES):
    """Convierte los artículos descargados en un ``DataFrame`` ``text;class``.

    Decisiones:

    * Artículos de la **categoría** del deporte: se guardan todos sus párrafos
      (heredan la etiqueta del artículo, con el ruido que eso supone).
    * Artículos del **buscador** (p. ej. "Juegos Olímpicos de 1920", la
      biografía de alguien que también jugó al rugby): solo se guardan los
      párrafos que contienen una palabra clave del deporte.
    * Un párrafo que menciona palabras clave de **dos deportes** se descarta.
    * Párrafos duplicados se eliminan (plantillas repetidas, o el mismo párrafo
      en dos clases).
    * Se guarda el título de origen y la fuente (columnas ``title`` y
      ``source``) para hacer particiones por artículo y analizar el ruido; el
      CSV del contrato solo contiene ``text;class``.
    """
    topics = topics or TOPICS
    kw = {label: re.compile(cfg["keywords"], re.IGNORECASE) for label, cfg in topics.items()}

    rows = []
    for label, arts in raw.items():
        for a in arts:
            if not a.get("text") or a["title"] in exclude_titles:
                continue
            source = a.get("source", "kategorio")
            for p in split_paragraphs(a["text"]):
                c = clean_paragraph(p, min_words, max_words)
                if not c:
                    continue
                mentions = {lab for lab, rx in kw.items() if rx.search(c)}
                if source != "kategorio" and label not in mentions:
                    continue            # párrafo del buscador que no habla del deporte
                if mentions - {label}:
                    continue            # menciona otro de los deportes: ambiguo
                rows.append({"text": c, "class": label, "title": a["title"], "source": source})
    df = pd.DataFrame(rows, columns=["text", "class", "title", "source"])
    df = df.drop_duplicates(subset="text", keep=False).reset_index(drop=True)
    return df


def save_dataset(df, path):
    """Guarda el dataset cumpliendo el contrato de datos."""
    assert not df["text"].str.contains(";").any()
    df[["text", "class"]].to_csv(path, sep=";", index=False, encoding="utf-8")


def load_dataset(path_or_url):
    """Carga un CSV con el contrato ``text;class``."""
    return pd.read_csv(path_or_url, sep=";", encoding="utf-8")
