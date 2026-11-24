"""Léxico cerrado del Esperanto.

El Esperanto tiene una gramática regular: casi todas las palabras "abiertas"
(sustantivos, adjetivos, verbos y adverbios derivados) se reconocen por su
terminación. Las excepciones son las palabras de clase cerrada (artículo,
pronombres, correlativos, preposiciones, conjunciones, numerales, adverbios
primarios...). Este módulo las enumera para que el etiquetador las reconozca
antes de aplicar las reglas de terminaciones.

Fuente: gramática básica del Esperanto (Fundamento de Esperanto, 16 reglas)
y la guía de la asignatura (nlp-esperantilo).
"""

# --------------------------------------------------------------------------
# Artículo
# --------------------------------------------------------------------------
ARTICLES = {"la", "l'"}

# --------------------------------------------------------------------------
# Pronombres personales (y sus formas en acusativo) y posesivos
# --------------------------------------------------------------------------
PERSONAL_PRONOUNS = ["mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si", "ci"]

# --------------------------------------------------------------------------
# Correlativos: tabla de 5 prefijos x 9 sufijos
#   ki- (interrogativo), ti- (demostrativo), i- (indefinido),
#   ĉi- (universal), neni- (negativo)
# --------------------------------------------------------------------------
CORRELATIVE_PREFIXES = ["ki", "ti", "i", "ĉi", "neni"]
# sufijo -> (categoría UPOS, significado)
CORRELATIVE_SUFFIXES = {
    "u": ("DET", "individuo"),     # kiu, tiu, iu, ĉiu, neniu  (declinable: -j, -n)
    "o": ("PRON", "cosa"),         # kio, tio, io, ĉio, nenio   (declinable: -n)
    "a": ("DET", "cualidad"),      # kia, tia, ...              (declinable: -j, -n)
    "e": ("ADV", "lugar"),         # kie, tie, ...              (-n = dirección)
    "am": ("ADV", "tiempo"),       # kiam, tiam, ...
    "el": ("ADV", "modo"),         # kiel, tiel, ...
    "om": ("ADV", "cantidad"),     # kiom, tiom, ...
    "al": ("ADV", "causa"),        # kial, tial, ...
    "es": ("DET", "posesión"),     # kies, ties, ...
}

PREPOSITIONS = {
    "al", "anstataŭ", "antaŭ", "apud", "da", "de", "dum", "ekster", "el", "en",
    "ĝis", "inter", "je", "kontraŭ", "krom", "kun", "laŭ", "malgraŭ", "per",
    "po", "por", "post", "preter", "pri", "pro", "sen", "sub", "super", "sur",
    "tra", "trans", "ĉe", "ĉirkaŭ", "cis", "kontraŭe", "fare",
}

CONJUNCTIONS = {
    "kaj", "aŭ", "sed", "nek", "do", "ĉar", "ke", "se", "kvankam", "ĉu",
    "kvazaŭ", "dum", "ol", "tamen", "ju", "des", "nome", "plus", "minus",
}

# Adverbios primarios (no terminan en -e de forma regular)
PRIMARY_ADVERBS = {
    "ankaŭ", "ankoraŭ", "apenaŭ", "baldaŭ", "eĉ", "for", "hieraŭ", "hodiaŭ",
    "jam", "ja", "jen", "morgaŭ", "ne", "jes", "nun", "nur", "plu", "pli",
    "plej", "preskaŭ", "tre", "tro", "tuj", "ĉi", "almenaŭ", "adiaŭ", "mem",
    "ajn", "ambaŭ", "kvazaŭ", "malpli", "malplej", "ĉiam", "neniam", "iom",
    "tiel", "kiel", "ankoraŭfoje",
}

PARTICLES = {"ajn", "ĉi", "ja", "jen", "mem", "nur", "eĉ"}

INTERJECTIONS = {"ho", "ha", "ve", "nu", "fi", "hura", "aĥ", "ehe", "hej", "bis", "saluton", "dankon"}

# Abreviaturas frecuentes: ktp (kaj tiel plu = etc.), ekz (ekzemple), ks (kaj simile)...
ABBREVIATIONS = {"ktp", "ekz", "ks", "kp", "vd", "ĉ", "pp", "ktr", "resp", "bv", "sk"}

# Numerales cardinales básicos
NUMERAL_UNITS = ["unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ"]
NUMERAL_WORDS = set(NUMERAL_UNITS) | {"nul", "dek", "cent", "mil"}

# Verbos auxiliares / muy frecuentes que se tratan como stop-words por lema
AUXILIARY_VERBS = {"esti", "havi", "povi", "devi", "voli", "fari", "iĝi", "igi", "ŝajni"}


def correlatives():
    """Genera la tabla completa de correlativos con sus formas declinadas.

    Devuelve un diccionario ``forma -> (upos, lema, rasgos)``.
    """
    table = {}
    for pre in CORRELATIVE_PREFIXES:
        for suf, (upos, _meaning) in CORRELATIVE_SUFFIXES.items():
            base = pre + suf
            table[base] = (upos, base, {"PronType": _pron_type(pre)})
            if suf in ("u", "a"):
                table[base + "j"] = (upos, base, {"PronType": _pron_type(pre), "Number": "Plur"})
                table[base + "n"] = (upos, base, {"PronType": _pron_type(pre), "Case": "Acc"})
                table[base + "jn"] = (upos, base, {"PronType": _pron_type(pre), "Number": "Plur", "Case": "Acc"})
            elif suf == "o":
                table[base + "n"] = (upos, base, {"PronType": _pron_type(pre), "Case": "Acc"})
            elif suf == "e":
                table[base + "n"] = (upos, base, {"PronType": _pron_type(pre), "Case": "Acc"})
    return table


def _pron_type(prefix):
    return {"ki": "Int", "ti": "Dem", "i": "Ind", "ĉi": "Tot", "neni": "Neg"}[prefix]


def pronouns():
    """Pronombres personales, posesivos y sus formas declinadas.

    Devuelve un diccionario ``forma -> (upos, lema, rasgos)``.
    """
    table = {}
    for p in PERSONAL_PRONOUNS:
        table[p] = ("PRON", p, {"PronType": "Prs"})
        table[p + "n"] = ("PRON", p, {"PronType": "Prs", "Case": "Acc"})
        poss = p + "a"  # mia, via, lia, ŝia, ĝia, nia, ilia, onia, sia
        table[poss] = ("DET", poss, {"Poss": "Yes"})
        table[poss + "j"] = ("DET", poss, {"Poss": "Yes", "Number": "Plur"})
        table[poss + "n"] = ("DET", poss, {"Poss": "Yes", "Case": "Acc"})
        table[poss + "jn"] = ("DET", poss, {"Poss": "Yes", "Number": "Plur", "Case": "Acc"})
    return table


def closed_class_lexicon():
    """Construye el léxico completo de palabras de clase cerrada.

    El orden importa: las entradas posteriores sobrescriben a las anteriores
    (p. ej. ``ĉi`` es a la vez prefijo correlativo y partícula).
    Devuelve ``forma -> (upos, lema, rasgos)``.
    """
    lex = {}
    for w in NUMERAL_WORDS:
        lex[w] = ("NUM", w, {"NumType": "Card"})
    for w in INTERJECTIONS:
        lex[w] = ("INTJ", w, {})
    for w in PRIMARY_ADVERBS:
        lex[w] = ("ADV", w, {})
    for w in ABBREVIATIONS:
        lex[w] = ("X", w, {"Abbr": "Yes"})
    for w in CONJUNCTIONS:
        upos = "SCONJ" if w in {"ke", "se", "ĉar", "kvankam", "ĉu", "kvazaŭ", "dum", "ol"} else "CCONJ"
        lex[w] = (upos, w, {})
    for w in PREPOSITIONS:
        lex[w] = ("ADP", w, {})
    lex.update(correlatives())
    lex.update(pronouns())
    for w in PARTICLES:
        lex[w] = ("PART", w, {})
    for w in ARTICLES:
        lex[w] = ("DET", "la", {"Definite": "Def", "PronType": "Art"})
    # "dum" es preposición y conjunción; dejamos ADP como más frecuente.
    lex["dum"] = ("ADP", "dum", {})
    return lex


# --------------------------------------------------------------------------
# Stop-words
# --------------------------------------------------------------------------
def default_stop_words():
    """Lista de stop-words: todas las palabras de clase cerrada + verbos auxiliares.

    Se guarda como conjunto de **lemas**, de forma que cualquier forma
    flexionada (``estas``, ``estis``, ``tiujn``...) se considera stop-word.
    """
    stops = {lemma for (_upos, lemma, _f) in closed_class_lexicon().values()}
    stops |= set(closed_class_lexicon())
    stops |= AUXILIARY_VERBS
    # palabras muy frecuentes sin carga temática en textos enciclopédicos
    stops |= {
        "multa", "multe", "pli", "alia", "sama", "nova", "unua", "dua", "tria",
        "granda", "malgranda", "bona", "nomi", "uzi", "ekzisti", "trovi",
        "jaro", "parto", "ekzemplo", "ekzemple", "speco", "tipo", "kelka",
        "diversa", "simila", "grava", "ofte", "kutime", "ĝenerale", "nomo",
    }
    return stops
