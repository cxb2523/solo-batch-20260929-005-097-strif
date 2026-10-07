from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import NamedTuple

__all__ = [
    "Insertion",
    "insert_multiple",
    "Replacement",
    "replace_multiple",
    "ShortIDCollisionError",
    "FoldError",
    "Folding",
    "short_id",
    "unique_short_id",
    "fold_repeated_substrings",
    "expand_repeated_substrings",
]

# Chosen from devtools/id_collision_study.py (seed 20260929):
# with 8 base36 chars the space is ~2.8e12, so at 100k distinct strings the
# birthday collision probability is ~1.8e-3 and no salted retry was ever
# needed, while 4 chars already collided in abundance and needed retries.
DEFAULT_SHORT_ID_LENGTH = 8
# Salted retries stay at zero up to 10k strings for lengths >= 6, but a
# pathological/adversarial run (or an exhausted short-ID space) must not loop
# forever: beyond this many distinct salts we treat the space as saturated.
DEFAULT_MAX_ID_ATTEMPTS = 32

_BASE36_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
_DEFAULT_OPEN_MARKER = "\u27e6"  # ⟦
_DEFAULT_CLOSE_MARKER = "\u27e7"  # ⟧


class Insertion(NamedTuple):
    offset: int
    text: str


def insert_multiple(text: str, insertions: list[Insertion]) -> str:
    """
    Insert multiple strings into `text` at the given offsets, at once.
    """
    chunks: list[str] = []
    last_end = 0
    for offset, insertion in sorted(insertions, key=lambda x: x[0]):
        chunks.append(text[last_end:offset])
        chunks.append(insertion)
        last_end = offset
    chunks.append(text[last_end:])
    return "".join(chunks)


class Replacement(NamedTuple):
    start: int
    end: int
    text: str


def replace_multiple(text: str, replacements: list[Replacement]) -> str:
    """
    Replace multiple substrings in `text` with new strings, simultaneously.
    The replacements are a list of tuples (start_offset, end_offset, new_string).
    """
    replacements = sorted(replacements, key=lambda x: x[0])
    chunks: list[str] = []
    last_end = 0
    for start, end, new_text in replacements:
        if start < last_end:
            raise ValueError("Overlapping replacements are not allowed.")
        chunks.append(text[last_end:start])
        chunks.append(new_text)
        last_end = end
    chunks.append(text[last_end:])
    return "".join(chunks)


class ShortIDCollisionError(RuntimeError):
    """
    Raised when no collision-free short ID could be assigned within the
    configured number of salted retries (the ID space is effectively full).
    """


class FoldError(ValueError):
    """
    Raised when folding/expanding references is impossible, e.g. the input
    already contains a literal reference marker or an unknown reference is
    expanded.
    """


def _base36_sha1(text: str, salt: int) -> str:
    payload = text if salt == 0 else f"{salt}\x00{text}"
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    number = int.from_bytes(digest, byteorder="big")
    chars: list[str] = []
    while number > 0:
        number, remainder = divmod(number, 36)
        chars.append(_BASE36_CHARS[remainder])
    return "".join(reversed(chars))


def short_id(text: str, length: int = DEFAULT_SHORT_ID_LENGTH, attempt: int = 0) -> str:
    """
    Deterministic short base36 ID for `text`: the first `length` characters of
    the base36 SHA1 digest, salted by `attempt` (0 = unsalted). Truncating the
    digest is what saves space but introduces birthday collisions, so callers
    assigning IDs to a table should use `unique_short_id`, which retries with
    successive salts on collision.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    return _base36_sha1(text, attempt)[:length].rjust(length, "0")


def unique_short_id(
    text: str,
    used_ids: set[str],
    length: int = DEFAULT_SHORT_ID_LENGTH,
    max_attempts: int = DEFAULT_MAX_ID_ATTEMPTS,
    id_func: Callable[[str, int, int], str] | None = None,
) -> str:
    """
    Return an ID for `text` that is not already in `used_ids`, retrying with
    salted hashes on collision. Mutates `used_ids` by inserting the result.

    The fallback (attempt >= 1) is the live collision path: it is exercised as
    soon as two texts share a truncated digest, and it raises
    `ShortIDCollisionError` once `max_attempts` distinct salts all collide,
    which signals a saturated ID space rather than dead code.

    `id_func(text, length, attempt)` may be injected (mainly for tests) to
    force collisions deterministically; it defaults to `short_id`.
    """
    generate = id_func or (lambda value, size, attempt: short_id(value, size, attempt))
    for attempt in range(max_attempts):
        candidate = generate(text, length, attempt)
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
    raise ShortIDCollisionError(
        f"could not assign a unique {length}-char id for {text!r} after {max_attempts} attempts"
    )


class Folding(NamedTuple):
    text: str
    ids: dict[str, str]


def _find_repeated_spans(text: str, min_len: int, max_len: int) -> list[tuple[int, int, str]]:
    # Map each repeated (>=2 occurrences) substring to its *first* span. Only
    # substrings that actually repeat are indexed, keeping this small for the
    # short, template-like inputs folding targets.
    first_span: dict[str, tuple[int, int]] = {}
    repeated: set[str] = set()
    limit = min(max_len, len(text))
    for size in range(min_len, limit + 1):
        for start in range(0, len(text) - size + 1):
            piece = text[start : start + size]
            if piece in first_span:
                repeated.add(piece)
            else:
                first_span[piece] = (start, start + size)

    # Second pass, scanning left to right, greedily takes the longest repeated
    # substring starting at each position. The first occurrence stays literal;
    # every later occurrence is a fold candidate, provided it does not overlap a
    # span already taken (overlaps cannot be emitted simultaneously).
    replacements: list[tuple[int, int, str]] = []
    occupied_until = 0
    for pos in range(len(text)):
        if pos < occupied_until:
            continue
        chosen: str | None = None
        upper = min(max_len, len(text) - pos)
        for size in range(upper, min_len - 1, -1):
            piece = text[pos : pos + size]
            first_start = first_span.get(piece)
            # Fold only when this exact substring already appeared strictly
            # earlier, so the kept literal occurrence unambiguously defines it.
            if first_start is not None and first_start[0] < pos:
                chosen = piece
                break
        if chosen is not None:
            end = pos + len(chosen)
            replacements.append((pos, end, chosen))
            occupied_until = end
    return replacements


def fold_repeated_substrings(
    text: str,
    min_length: int = 2,
    max_length: int = 64,
    id_length: int = DEFAULT_SHORT_ID_LENGTH,
    max_id_attempts: int = DEFAULT_MAX_ID_ATTEMPTS,
    open_marker: str = _DEFAULT_OPEN_MARKER,
    close_marker: str = _DEFAULT_CLOSE_MARKER,
    id_func: Callable[[str, int, int], str] | None = None,
) -> Folding:
    """
    Fold repeated substrings of `text` into short reference IDs.

    The first occurrence of every repeated substring is kept verbatim; later
    non-overlapping occurrences are replaced with `⟦id⟧`. Returns the folded
    text plus an `ids` mapping from reference ID to the original substring.
    Use `expand_repeated_substrings` to reverse this.

    Failure branches:
    - empty/short text or no repeats -> folded text equals the input and
      `ids` is empty,
    - a literal marker already present in `text` -> `FoldError`, since emitted
      references would be ambiguous,
    - no unique ID within `max_id_attempts` salts -> `ShortIDCollisionError`.
    """
    if not open_marker or not close_marker:
        raise ValueError("markers must be non-empty")
    if open_marker in text or close_marker in text:
        raise FoldError("input already contains a reference marker; pass different markers")
    if not text or len(text) < min_length:
        return Folding(text, {})

    spans = _find_repeated_spans(text, min_length, max_length)
    ids: dict[str, str] = {}
    used_ids: set[str] = set()
    replacements: list[Replacement] = []
    for start, end, phrase in spans:
        ref = ids.get(phrase)
        if ref is None:
            ref = unique_short_id(
                phrase,
                used_ids,
                length=id_length,
                max_attempts=max_id_attempts,
                id_func=id_func,
            )
            ids[phrase] = ref
        replacements.append(Replacement(start, end, f"{open_marker}{ref}{close_marker}"))

    folded = replace_multiple(text, replacements)
    ref_ids = {ref: phrase for phrase, ref in ids.items()}
    return Folding(folded, ref_ids)


def expand_repeated_substrings(
    text: str,
    ids: dict[str, str],
    open_marker: str = _DEFAULT_OPEN_MARKER,
    close_marker: str = _DEFAULT_CLOSE_MARKER,
) -> str:
    """
    Reverse `fold_repeated_substrings`: replace every `⟦id⟧` reference with the
    substring from `ids`. Raises `FoldError` on an unknown reference.
    """
    result: list[str] = []
    pos = 0
    while True:
        start = text.find(open_marker, pos)
        if start == -1:
            result.append(text[pos:])
            break
        end = text.find(close_marker, start + len(open_marker))
        if end == -1:
            raise FoldError("unterminated reference marker")
        ref = text[start + len(open_marker) : end]
        if ref not in ids:
            raise FoldError(f"unknown reference id: {ref!r}")
        result.append(text[pos:start])
        result.append(ids[ref])
        pos = end + len(close_marker)
    return "".join(result)
