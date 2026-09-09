"""Identity functions.

These are the most expensive decisions in the system to reverse, so they live
alone in one small, heavily tested module.

Three rules, each of which exists because violating it breaks something specific:

1. ``chunk_uid`` hashes *coordinates*, never chunk text. Contextual retrieval
   prepends LLM-generated text to a chunk; if the id hashed the text, ids would
   be nondeterministic across reruns, spurious tombstones would fire, and
   blue/green index diffing would break.
2. ``content_hash`` is separate and is used only for deduplication and for the
   embedding cache key.
3. Documents are keyed by *version-stripped* arXiv base id. Five versions of one
   paper are one document; otherwise top-k fills with near-duplicates and
   recall@k collapses for reasons unrelated to the retriever.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

_SEP = "\x1f"  # ASCII unit separator: cannot occur in ids or config names

# New-style: 2401.01234 / 2401.01234v3  (4-digit YYMM, 4-or-5-digit sequence)
_ARXIV_NEW = re.compile(r"^(?P<base>\d{4}\.\d{4,5})(?:v(?P<version>\d+))?$")
# Old-style: hep-th/9901001 / math.GT/0309136v2
_ARXIV_OLD = re.compile(
    r"^(?P<base>[a-z][a-z\-\.]*(?:\.[A-Z]{2})?/\d{7})(?:v(?P<version>\d+))?$"
)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_arxiv_id(raw: str) -> tuple[str, int | None]:
    """Split an arXiv identifier into (base_id, version).

    Accepts bare ids, ``arXiv:`` prefixes, and full abs/pdf URLs.

    >>> canonical_arxiv_id("arXiv:2401.01234v3")
    ('2401.01234', 3)
    >>> canonical_arxiv_id("https://arxiv.org/abs/hep-th/9901001")
    ('hep-th/9901001', None)

    Raises:
        ValueError: if the string is not a recognisable arXiv identifier. We
            raise rather than returning a sentinel because a silently
            mis-parsed id would corrupt the document key space.
    """
    s = raw.strip()
    s = re.sub(r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = re.sub(r"^arxiv:", "", s, flags=re.I)

    for pattern in (_ARXIV_NEW, _ARXIV_OLD):
        m = pattern.match(s)
        if m:
            version = m.group("version")
            return m.group("base"), (int(version) if version else None)

    raise ValueError(f"not a recognisable arXiv identifier: {raw!r}")


def normalize_text(text: str) -> str:
    """Normalisation applied once, before the canonical text is frozen.

    Deliberately conservative: NFC, CRLF/CR -> LF, tabs -> spaces, trailing
    whitespace stripped per line. It must never change character *counts* in a
    way that would silently shift stored offsets after the fact, so it is
    applied exactly once at ingest and the result is hashed.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def content_hash(text: str) -> str:
    """Hash of already-normalised text. Used for dedup and the embedding cache."""
    return _sha256(text)


def chunk_uid(doc_id: str, chunker_config_id: str, char_start: int, char_end: int) -> str:
    """Stable chunk identity: coordinates, never content.

    Raises:
        ValueError: on an empty or inverted span, which would otherwise produce
            a valid-looking id for a chunk that cannot be resolved back to text.
    """
    if char_start < 0:
        raise ValueError(f"char_start must be >= 0, got {char_start}")
    if char_end <= char_start:
        raise ValueError(f"empty or inverted span: [{char_start}, {char_end})")
    if _SEP in doc_id or _SEP in chunker_config_id:
        raise ValueError("doc_id and chunker_config_id must not contain \x1f")
    return _sha256(_SEP.join([doc_id, chunker_config_id, str(char_start), str(char_end)]))


def _canonical(obj: Any) -> Any:
    """Recursively canonicalise for hashing: dict key order must not matter."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def config_hash(config: Any, length: int = 16) -> str:
    """Hash a config/manifest into a short stable run id.

    Key order and int/float spelling are normalised so that a semantically
    identical config always produces the same id.
    """
    payload = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(payload)[:length]
