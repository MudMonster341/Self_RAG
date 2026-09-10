"""Corpus acquisition and parsing.

The contract that binds every module in this package together, and the one that
is expensive to get wrong:

**A parser's final step is ``normalize_text()``, and every offset it reports is
an index into that normalised string.**

Normalisation is not length-preserving -- CRLF collapses to LF, trailing
whitespace is stripped -- so offsets computed against raw text and then
normalised are silently wrong by a drifting amount. Since qrels, citation spans
and ``chunk_uid`` are all coordinates into the canonical text, a drift of even
one character is unrecoverable once judgements exist.

Enforced by an invariant test rather than by convention:
``normalize_text(doc.text) == doc.text`` for every parser output, and every
reported section span must slice to non-empty text.
"""
