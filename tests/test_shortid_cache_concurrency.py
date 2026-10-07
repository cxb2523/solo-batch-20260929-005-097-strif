import threading

import pytest

from strif import (
    AtomicUpdateConflict,
    AtomicVar,
    FoldError,
    Folding,
    ShortIDCollisionError,
    StringTemplate,
    expand_repeated_substrings,
    fold_repeated_substrings,
    short_id,
    unique_short_id,
)
from strif.string_template import TemplateCache, _plans_match

# ---- short id: deterministic generation and the salted fallback -------------


def test_short_id_is_deterministic_and_salted():
    assert short_id("abc", length=8) == short_id("abc", length=8)
    assert len(short_id("abc", length=8)) == 8
    # The salted fallback produces different ids for different attempts.
    assert short_id("abc", length=8, attempt=0) != short_id("abc", length=8, attempt=1)


def test_short_id_rejects_bad_length_and_attempt():
    with pytest.raises(ValueError):
        short_id("x", length=0)
    with pytest.raises(ValueError):
        short_id("x", attempt=-1)


def test_unique_short_id_avoids_live_collisions():
    # The unsalted value is already taken, so the first call must fall back.
    used: set[str] = {"dup"}
    seq = {
        ("a", 0): "dup",
        ("a", 1): "a1",
        ("b", 0): "dup",
        ("b", 1): "b1",
    }

    def id_func(text, length, attempt):
        return seq[(text, attempt)]

    first = unique_short_id("a", used, id_func=id_func)
    second = unique_short_id("b", used, id_func=id_func)
    assert first == "a1"
    assert second == "b1"
    assert {"a1", "b1", "dup"} <= used


def test_unique_short_id_fallback_is_executed_not_dead_code():
    used: set[str] = {"fixed"}

    def id_func(text, length, attempt):
        # Unsalted choice always collides; only a salted attempt succeeds.
        return "fixed" if attempt == 0 else f"salt-{attempt}"

    result = unique_short_id("anything", used, id_func=id_func)
    assert result == "salt-1"
    assert "salt-1" in used


def test_unique_short_id_raises_when_space_saturates():
    used: set[str] = {"always"}

    def id_func(text, length, attempt):
        return "always"

    with pytest.raises(ShortIDCollisionError):
        unique_short_id("x", used, max_attempts=3, id_func=id_func)


# ---- repeated substring folding ---------------------------------------------


def test_fold_empty_template_like_text():
    result = fold_repeated_substrings("")
    assert isinstance(result, Folding)
    assert result.text == ""
    assert result.ids == {}
    assert expand_repeated_substrings(result.text, result.ids) == ""


def test_fold_without_repeats_is_identity():
    result = fold_repeated_substrings("abc")
    assert result.text == "abc"
    assert result.ids == {}


def test_fold_repeated_placeholder_collapses_to_one_reference():
    # Single-use separators keep "@name" as the maximal repeated unit, so every
    # repeated placeholder folds to a single shared reference.
    text = "@name#@name$@name%"
    result = fold_repeated_substrings(text)
    assert len(result.ids) == 1
    assert result.text.count(next(iter(result.ids))) == 2
    assert expand_repeated_substrings(result.text, result.ids) == text


def test_fold_roundtrip_several_phrases():
    text = "alpha beta alpha beta gamma beta"
    result = fold_repeated_substrings(text)
    assert expand_repeated_substrings(result.text, result.ids) == text


def test_fold_collision_fallback_assigns_distinct_ids():
    # Single-use separators ensure each token (not a longer compound) is the
    # repeated unit, so exactly three ids are assigned.
    text = "abcpdefqghirabcsdeftghi"

    def id_func(value, length, attempt):
        # Unsalted hash collides for the first two tokens; salts disambiguate.
        table = {
            ("abc", 0): "same",
            ("abc", 1): "salt-a-1",
            ("def", 0): "same",
            ("def", 1): "salt-d-1",
            ("ghi", 0): "id-ghi",
        }
        return table[(value, attempt)]

    result = fold_repeated_substrings(text, id_func=id_func)
    assert len(result.ids) == 3
    assert len(set(result.ids)) == 3
    assert expand_repeated_substrings(result.text, result.ids) == text


def test_fold_rejects_literal_marker_in_input():
    with pytest.raises(FoldError):
        fold_repeated_substrings("already has a \u27e6 marker")


def test_expand_unknown_reference_raises():
    with pytest.raises(FoldError):
        expand_repeated_substrings("see \u27e6nope\u27e7", {})
    with pytest.raises(FoldError):
        expand_repeated_substrings("unterminated \u27e6xyz", {})


# ---- string template cache ---------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_template_cache():
    StringTemplate.cache.invalidate()
    yield
    StringTemplate.cache.invalidate()


def test_empty_template_formats_to_empty_and_caches():
    template = StringTemplate("", [])
    assert template.format() == ""
    assert not template
    assert template.format() == ""
    assert StringTemplate.cache.hits >= 1


def test_repeated_placeholder_formats_each_occurrence():
    template = StringTemplate("{x}|{x}|{x}", ["x"])
    assert template.format(x=7) == "7|7|7"


def test_cache_isolates_distinct_templates():
    a = StringTemplate("{a}", ["a"])
    b = StringTemplate("{b}", ["b"])
    assert a.format(a="A") == "A"
    assert b.format(b="B") == "B"
    assert len(StringTemplate.cache) >= 2


def test_cache_invalidation_forces_recompute():
    template = StringTemplate("{a}", ["a"])
    template.format(a="x")
    StringTemplate.invalidate_cache("{a}")
    assert StringTemplate.cache.invalidations >= 1
    assert template.format(a="y") == "y"


def test_stale_cache_entry_is_invalidated_and_recomputed(monkeypatch):
    template = StringTemplate("{a}", ["a"])
    field_types = template._field_types()
    template._get_plans(field_types)

    # Poison the cache with a plan that cannot rebuild back to this template:
    # a failed hit must be dropped and recomputed rather than render garbage.
    other = TemplateCache._parse("{totally_different}")
    monkeypatch.setitem(
        StringTemplate.cache._entries,
        TemplateCache._key(template.template, field_types),
        other,
    )
    assert not _plans_match(other, template.template)
    invalidations_before = StringTemplate.cache.invalidations
    assert template.format(a="value") == "value"
    assert StringTemplate.cache.invalidations > invalidations_before
    assert _plans_match(template._get_plans(field_types), template.template)


# ---- atomic var: concurrency -------------------------------------------------


def test_compare_and_swap_commits_or_rejects():
    var = AtomicVar(0)
    _, version = var.snapshot()
    assert var.compare_and_swap(version, 5) is True
    assert var.value == 5
    # The stale version token must now fail rather than overwrite.
    assert var.compare_and_swap(version, 99) is False
    assert var.value == 5


def test_retry_update_never_loses_concurrent_updates():
    var = AtomicVar(0)
    threads_count = 12
    increments = 2000
    barrier = threading.Barrier(threads_count)

    def worker():
        barrier.wait()
        for _ in range(increments):
            var.retry_update(lambda value: value + 1)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert var.value == threads_count * increments
    assert var.version == threads_count * increments


def test_retry_update_retries_on_conflict_then_succeeds(monkeypatch):
    var = AtomicVar(10)
    state = {"interfere": True}
    real_cas = var.compare_and_swap

    def flaky_cas(expected_version, new_value):
        # A concurrent writer wins the first race, forcing a genuine retry.
        if state["interfere"]:
            state["interfere"] = False
            var.set(var.value + 100)
        return real_cas(expected_version, new_value)

    monkeypatch.setattr(var, "compare_and_swap", flaky_cas)
    result = var.retry_update(lambda value: value + 1, max_attempts=10)
    assert result == var.value
    assert var.value == 111  # 10 + interfering 100 + retried 1


def test_retry_update_raises_after_exhausting_attempts(monkeypatch):
    var = AtomicVar(0)
    monkeypatch.setattr(var, "compare_and_swap", lambda version, value: False)
    with pytest.raises(AtomicUpdateConflict):
        var.retry_update(lambda value: value + 1, max_attempts=3)
    with pytest.raises(ValueError):
        var.retry_update(lambda value: value, max_attempts=0)
