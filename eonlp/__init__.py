"""eonlp - NLP basado en reglas y clasificación temática de textos en Esperanto.

Práctica de Minería de Textos (2026/2027).

    >>> import eonlp
    >>> nlp = eonlp.load()
    >>> [t.lemma_ for t in nlp("La katoj manĝis.")]
    ['la', 'kato', 'manĝi', '.']
"""

from .nlp import Doc, Esperanto, Span, Token, load, normalize_text, split_morphemes

__version__ = "1.0.0"
__all__ = ["Doc", "Esperanto", "Span", "Token", "load", "normalize_text", "split_morphemes"]
