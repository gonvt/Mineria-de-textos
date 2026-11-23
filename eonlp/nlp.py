"""Modelo de NLP para Esperanto basado en reglas, con un API similar a spaCy.

Uso::

    nlp = load()
    doc = nlp("La katoj manĝis fiŝon.")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)

Componentes del pipeline (``nlp.pipe_names``):

1. ``normalizer``  - normaliza Unicode y el sistema-x (``cx`` -> ``ĉ``).
2. ``tokenizer``   - divide el texto en tokens con expresiones regulares.
3. ``tagger``      - etiquetado gramatical (UPOS) y morfológico por terminaciones.
4. ``lemmatizer``  - forma de diccionario (``-o``, ``-a``, ``-i``, ``-e``).
5. ``stopwords``   - marca ``token.is_stop`` a partir de una lista de lemas.

Todo está implementado desde cero, sin librerías de NLP: el Esperanto es
regular y sin excepciones, así que unas pocas reglas cubren la mayoría de casos.
"""

import re
import unicodedata

from .lexicon import NUMERAL_UNITS, closed_class_lexicon, default_stop_words

# --------------------------------------------------------------------------
# Normalización
# --------------------------------------------------------------------------
# Sistema-x: forma habitual de escribir Esperanto sin teclado especial.
X_SYSTEM = {
    "cx": "ĉ", "gx": "ĝ", "hx": "ĥ", "jx": "ĵ", "sx": "ŝ", "ux": "ŭ",
    "Cx": "Ĉ", "Gx": "Ĝ", "Hx": "Ĥ", "Jx": "Ĵ", "Sx": "Ŝ", "Ux": "Ŭ",
    "CX": "Ĉ", "GX": "Ĝ", "HX": "Ĥ", "JX": "Ĵ", "SX": "Ŝ", "UX": "Ŭ",
}
_X_RE = re.compile("|".join(X_SYSTEM))


# Variantes ortográficas con "ĥ" que hoy se escriben con "k"
# (ĥemio -> kemio, teĥniko -> tekniko...). Se unifican para que el modelo
# vea una sola forma de cada raíz.
H_VARIANTS = {"ĥem": "kem", "ĥlor": "klor", "ĥrom": "krom", "ĥaos": "kaos",
              "teĥn": "tekn", "meĥan": "mekan", "arĥe": "arke", "ĥiral": "kiral",
              "ĥoler": "koler", "ĥirurg": "kirurg"}
_H_RE = re.compile("|".join(H_VARIANTS), re.IGNORECASE)


def unify_spelling(word):
    """Unifica variantes ortográficas: ``bioĥemia`` -> ``biokemia``."""
    if "ĥ" not in word.lower():
        return word
    return _H_RE.sub(lambda m: H_VARIANTS[m.group(0).lower()], word)


def normalize_text(text, x_system=True):
    """Normaliza a NFC (letras con diacrítico en un solo carácter) y,
    opcionalmente, convierte el sistema-x a letras con diacrítico."""
    text = unicodedata.normalize("NFC", text)
    # comillas y apóstrofos tipográficos -> ASCII
    text = text.replace("’", "'").replace("‘", "'").replace("´", "'")
    if x_system:
        text = _X_RE.sub(lambda m: X_SYSTEM[m.group(0)], text)
    return text


# --------------------------------------------------------------------------
# Tokenizador
# --------------------------------------------------------------------------
ESPERANTO_LETTERS = "a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ"
_TOKEN_RE = re.compile(
    rf"""
    (?P<url>https?://\S+)                                   # URLs
  | (?P<num>\d+(?:[.,]\d+)*)                                # números: 1 000, 3,14
  | (?P<word>[\w]+(?:[-'][\w]+)*'?)                         # palabras (incl. guion y elisión l')
  | (?P<punct>[^\w\s])                                      # signos de puntuación
    """,
    re.VERBOSE | re.UNICODE,
)


def tokenize(text):
    """Devuelve una lista de tuplas ``(texto, posición, espacio_posterior)``."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        start, end = m.span()
        ws = text[end:end + 1] if end < len(text) and text[end].isspace() else ""
        out.append((m.group(0), start, ws))
    return out


# --------------------------------------------------------------------------
# Objetos Token y Doc (API tipo spaCy)
# --------------------------------------------------------------------------
class MorphAnalysis(dict):
    """Rasgos morfológicos (``Number=Plur|Case=Acc``) al estilo Universal Dependencies."""

    def __str__(self):
        return "|".join(f"{k}={v}" for k, v in sorted(self.items()))

    def get(self, key, default=None):  # noqa: D401 - igual que dict.get
        return super().get(key, default)


class Token:
    """Un token del documento.

    Atributos principales (mismos nombres que spaCy):
    ``text``, ``i``, ``idx``, ``whitespace_``, ``lemma_``, ``pos_``, ``tag_``,
    ``morph``, ``is_stop``, ``is_punct``, ``is_alpha``, ``like_num``, ``like_url``.
    Extra propio: ``root_`` (raíz sin terminación gramatical) y ``morphemes``.
    """

    __slots__ = ("doc", "text", "i", "idx", "whitespace_", "lemma_", "pos_",
                 "tag_", "morph", "is_stop", "root_", "morphemes")

    def __init__(self, doc, text, i, idx, whitespace=""):
        self.doc = doc
        self.text = text
        self.i = i
        self.idx = idx
        self.whitespace_ = whitespace
        self.lemma_ = text.lower()
        self.pos_ = "X"
        self.tag_ = ""
        self.morph = MorphAnalysis()
        self.is_stop = False
        self.root_ = text.lower()
        self.morphemes = [text.lower()]

    # propiedades derivadas -------------------------------------------------
    @property
    def lower_(self):
        return self.text.lower()

    @property
    def text_with_ws(self):
        return self.text + self.whitespace_

    @property
    def is_punct(self):
        return self.pos_ == "PUNCT"

    @property
    def is_alpha(self):
        return self.text.replace("'", "").isalpha()

    @property
    def is_digit(self):
        return self.text.isdigit()

    @property
    def like_num(self):
        return self.pos_ == "NUM"

    @property
    def like_url(self):
        return self.text.startswith(("http://", "https://"))

    @property
    def is_sent_start(self):
        return self.i == 0 or self.doc[self.i - 1].text in {".", "!", "?"}

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Span:
    """Secuencia contigua de tokens (se usa para las oraciones)."""

    def __init__(self, doc, start, end):
        self.doc, self.start, self.end = doc, start, end

    def __iter__(self):
        return iter(self.doc.tokens[self.start:self.end])

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        return self.doc.tokens[self.start:self.end][i]

    @property
    def text(self):
        return "".join(t.text_with_ws for t in self).strip()

    def __repr__(self):
        return self.text


class Doc:
    """Documento procesado: contenedor de ``Token`` en orden."""

    def __init__(self, text, words=None):
        self.text = text
        self.tokens = []
        for i, (w, idx, ws) in enumerate(words or []):
            self.tokens.append(Token(self, w, i, idx, ws))

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, _ = i.indices(len(self))
            return Span(self, start, stop)
        return self.tokens[i]

    @property
    def sents(self):
        """Segmentación en oraciones por signos de puntuación final."""
        start = 0
        for t in self.tokens:
            if t.text in {".", "!", "?"}:
                yield Span(self, start, t.i + 1)
                start = t.i + 1
        if start < len(self.tokens):
            yield Span(self, start, len(self.tokens))

    def to_table(self):
        """Lista de diccionarios (cómodo para crear un ``pandas.DataFrame``)."""
        return [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
             "morph": str(t.morph), "root": t.root_, "is_stop": t.is_stop}
            for t in self.tokens
        ]

    def __repr__(self):
        return self.text


# --------------------------------------------------------------------------
# Análisis morfológico por terminaciones
# --------------------------------------------------------------------------
# Terminaciones verbales: tiempo/modo
VERB_ENDINGS = {
    "as": {"Tense": "Pres", "Mood": "Ind", "VerbForm": "Fin"},
    "is": {"Tense": "Past", "Mood": "Ind", "VerbForm": "Fin"},
    "os": {"Tense": "Fut", "Mood": "Ind", "VerbForm": "Fin"},
    "us": {"Mood": "Cnd", "VerbForm": "Fin"},
    "u": {"Mood": "Imp", "VerbForm": "Fin"},
    "i": {"VerbForm": "Inf"},
}

# Sufijos de participio: (aspecto, tiempo, voz)
PARTICIPLES = {
    "ant": ("Imp", "Pres", "Act"), "int": ("Perf", "Past", "Act"), "ont": ("Prosp", "Fut", "Act"),
    "at": ("Imp", "Pres", "Pass"), "it": ("Perf", "Past", "Pass"), "ot": ("Prosp", "Fut", "Pass"),
}
# Palabras que parecen participios pero no lo son (raíz lexicalizada)
PARTICIPLE_EXCEPTIONS = {
    "konstanta", "konstante", "elefanta", "kvanta", "privata", "delikata", "akurata",
    "adekvata", "infinita", "eksplicita", "implicita", "favorita", "granda",
    "kandidata", "senata", "pirata", "kvalita", "klimata", "simpatia", "planta",
    "giganta", "elegante", "eleganta", "brilanta", "variante", "konsonanta",
    "elita", "spirita", "merita", "subita", "subite", "limita", "unuopa", "dekstra",
    "kalkulita", "sankta", "akcidenta", "frekventa", "inteligenta", "potenca",
    "kompetenta", "evidenta", "evidente", "permanenta", "permanente", "absoluta",
    "absolute", "pacienta", "studenta", "doktoranta", "abunda",
}
MIN_STEM = 2  # longitud mínima de raíz para aceptar una terminación

# Afijos (para descomponer en morfemas: mal-san-ul-ej-o)
PREFIXES = ["mal", "re", "ek", "dis", "ge", "eks", "bo", "fi", "mis", "pra", "for", "ne"]
SUFFIXES = ["ad", "aĉ", "aĵ", "an", "ar", "ec", "eg", "ej", "em", "end", "er", "estr",
            "et", "id", "ig", "iĝ", "il", "in", "ind", "ing", "ism", "ist", "obl", "on",
            "op", "uj", "ul", "um"]

_NUM_RE = re.compile(
    r"^(?:(?:" + "|".join(NUMERAL_UNITS) + r")?(?:dek|cent|mil))+(?:" + "|".join(NUMERAL_UNITS) + r")?$"
)
_FOREIGN_RE = re.compile(r"[qwxyQWXY]")


def split_morphemes(stem, min_root=3):
    """Descompone (de forma voraz) una raíz en prefijos + raíz + sufijos.

    ``malsanulejo`` -> raíz ``malsanulej`` -> ``['mal', 'san', 'ul', 'ej']``.
    Solo se separa un afijo si la raíz restante conserva ``min_root`` letras,
    para no destrozar raíces cortas (``ideo`` no es ``id-eo``).
    """
    prefixes, suffixes = [], []
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if stem.startswith(p) and len(stem) - len(p) >= min_root + 1:
                prefixes.append(p)
                stem = stem[len(p):]
                changed = True
                break
    changed = True
    while changed:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if stem.endswith(s) and len(stem) - len(s) >= min_root:
                suffixes.insert(0, s)
                stem = stem[: -len(s)]
                changed = True
                break
    return prefixes + [stem] + suffixes


class EsperantoAnalyzer:
    """Etiquetador + lematizador basado en reglas.

    Devuelve, para una palabra, ``(upos, lemma, rasgos, raíz)``.
    """

    def __init__(self):
        self.lexicon = closed_class_lexicon()

    def analyze(self, word, sent_start=False):
        low = unify_spelling(word.lower())

        # 1. Puntuación y números ------------------------------------------
        if not any(ch.isalnum() for ch in word):
            return "PUNCT", word, {}, word
        if word.replace(".", "").replace(",", "").isdigit():
            return "NUM", word, {"NumType": "Card"}, word
        if word.startswith(("http://", "https://")):
            return "X", word, {}, word

        # 2. Léxico cerrado -------------------------------------------------
        if low in self.lexicon:
            upos, lemma, feats = self.lexicon[low]
            return upos, lemma, dict(feats), lemma
        if _NUM_RE.match(low):
            return "NUM", low, {"NumType": "Card"}, low

        # 3. Elisión poética: "kor'" = "koro"
        if low.endswith("'") and len(low) > 2:
            return "NOUN", low[:-1] + "o", {"Number": "Sing", "Case": "Nom"}, low[:-1]

        # 4. Nombres propios y palabras extranjeras ------------------------
        is_capital = word[:1].isupper()
        if _FOREIGN_RE.search(low) or any(ch.isdigit() for ch in low):
            return ("PROPN" if is_capital else "X"), word, {"Foreign": "Yes"}, low
        if is_capital and not sent_start and not self._has_eo_ending(low):
            return "PROPN", word, {}, low

        # 5. Reglas de terminación (núcleo del modelo) ---------------------
        res = self._by_ending(low)
        if res is not None:
            upos, lemma, feats, root = res
            if is_capital and not sent_start:
                # En mitad de la oración, la mayúscula indica nombre propio
                # ("Eŭropo", "Parizo", "Marie Curie"). Los adjetivos de
                # nacionalidad se escriben en minúscula en Esperanto.
                upos = "PROPN"
                lemma = word[:1] + lemma[1:] if res[0] == "NOUN" else word
            return upos, lemma, feats, root

        return ("PROPN" if is_capital else "X"), word if is_capital else low, {}, low

    # ------------------------------------------------------------------
    @staticmethod
    def _has_eo_ending(low):
        return bool(re.search(r"(o|oj|on|ojn|a|aj|an|ajn|e|en|as|is|os|us|u|i)$", low))

    def _by_ending(self, low):
        # Sustantivos: -o(j)(n)
        m = re.match(r"^(.+?)o(j?)(n?)$", low)
        if m and len(m.group(1)) >= MIN_STEM:
            stem = m.group(1)
            feats = {"Number": "Plur" if m.group(2) else "Sing", "Case": "Acc" if m.group(3) else "Nom"}
            return "NOUN", stem + "o", feats, stem

        # Adjetivos: -a(j)(n)  (pueden ser participios: -ant-a, -it-a...)
        m = re.match(r"^(.+?)a(j?)(n?)$", low)
        if m and len(m.group(1)) >= MIN_STEM:
            stem = m.group(1)
            feats = {"Number": "Plur" if m.group(2) else "Sing", "Case": "Acc" if m.group(3) else "Nom"}
            part = self._participle(stem, low)
            if part:
                verb_stem, pfeats = part
                feats.update(pfeats)
                return "ADJ", verb_stem + "i", feats, verb_stem
            if stem in NUMERAL_UNITS or stem in {"dek", "cent", "mil"} or _NUM_RE.match(stem):
                feats["NumType"] = "Ord"     # dua, tria, dudeka...
            return "ADJ", stem + "a", feats, stem

        # Adverbios derivados: -e(n)
        m = re.match(r"^(.+?)e(n?)$", low)
        if m and len(m.group(1)) >= MIN_STEM:
            stem = m.group(1)
            feats = {"Case": "Acc"} if m.group(2) else {}   # -en = dirección (hejmen)
            part = self._participle(stem, low)
            if part:
                verb_stem, pfeats = part
                feats.update(pfeats)
                return "ADV", verb_stem + "i", feats, verb_stem
            return "ADV", stem + "e", feats, stem

        # Verbos: -as, -is, -os, -us, -u, -i
        for ending in ("as", "is", "os", "us", "u", "i"):
            if low.endswith(ending) and len(low) - len(ending) >= MIN_STEM:
                stem = low[: -len(ending)]
                return "VERB", stem + "i", dict(VERB_ENDINGS[ending]), stem
        return None

    @staticmethod
    def _participle(stem, word):
        """Si ``stem`` termina en un sufijo de participio, devuelve la raíz verbal."""
        if word in PARTICIPLE_EXCEPTIONS:
            return None
        for suf, (aspect, tense, voice) in PARTICIPLES.items():
            if stem.endswith(suf) and len(stem) - len(suf) >= 3:
                return stem[: -len(suf)], {"VerbForm": "Part", "Aspect": aspect,
                                            "Tense": tense, "Voice": voice}
        return None


# --------------------------------------------------------------------------
# Pipeline estilo spaCy
# --------------------------------------------------------------------------
class Esperanto:
    """Objeto ``nlp``: aplica el pipeline completo a un texto y devuelve un ``Doc``."""

    lang = "eo"

    def __init__(self, stop_words=None, x_system=True):
        self.x_system = x_system
        self.analyzer = EsperantoAnalyzer()
        self.stop_words = set(stop_words) if stop_words is not None else default_stop_words()
        self.pipeline = [
            ("tagger", self._tagger),
            ("lemmatizer", self._lemmatizer),
            ("stopwords", self._stopwords),
        ]

    @property
    def pipe_names(self):
        return ["normalizer", "tokenizer"] + [name for name, _ in self.pipeline]

    # API pública ---------------------------------------------------------
    def make_doc(self, text):
        """Normaliza y tokeniza (sin el resto del pipeline)."""
        text = normalize_text(text, self.x_system)
        return Doc(text, tokenize(text))

    def __call__(self, text):
        doc = self.make_doc(text)
        for _name, component in self.pipeline:
            doc = component(doc)
        return doc

    def pipe(self, texts):
        """Procesa un iterable de textos (generador, como ``spacy.Language.pipe``)."""
        for text in texts:
            yield self(text)

    def add_stop_words(self, words):
        self.stop_words |= {w.lower() for w in words}

    # Componentes ---------------------------------------------------------
    def _tagger(self, doc):
        for tok in doc:
            upos, lemma, feats, root = self.analyzer.analyze(tok.text, tok.is_sent_start)
            tok.pos_ = upos
            tok.morph = MorphAnalysis(feats)
            tok.tag_ = upos + ("__" + str(tok.morph) if feats else "")
            tok.lemma_ = lemma   # se guarda aquí y lo confirma el lematizador
            tok.root_ = root
        return doc

    def _lemmatizer(self, doc):
        for tok in doc:
            if tok.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}:
                tok.lemma_ = tok.lemma_.lower()
                tok.morphemes = split_morphemes(tok.root_)
            elif tok.pos_ == "PROPN":
                tok.morphemes = [tok.root_]
            else:
                tok.morphemes = [tok.lemma_.lower()]
        return doc

    def _stopwords(self, doc):
        for tok in doc:
            tok.is_stop = tok.lower_ in self.stop_words or tok.lemma_.lower() in self.stop_words
        return doc

    def __repr__(self):
        return f"<Esperanto nlp pipeline={self.pipe_names}>"


def load(**kwargs):
    """Equivalente a ``spacy.load(...)``: devuelve el pipeline listo para usar."""
    return Esperanto(**kwargs)
