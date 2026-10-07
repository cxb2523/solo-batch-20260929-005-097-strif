import threading
from collections.abc import Sequence
from dataclasses import dataclass
from string import Formatter
from typing import Any, ClassVar, NamedTuple

__all__ = ["StringTemplate", "TemplateCache", "TemplatePlan"]


class TemplatePlan(NamedTuple):
    """
    Parsed representation of a template: the literal pieces and field
    descriptors produced by ``string.Formatter.parse``. Rendering a cached plan
    avoids re-parsing the template on every ``format`` call.
    """

    literal: str
    field_name: str | None
    format_spec: str | None
    conversion: str | None


class TemplateCache:
    """
    Thread-safe cache of parsed ``StringTemplate`` plans.

    Entries are keyed by both the template text and the normalized allowed-field
    mapping, so plans are freely reused across identical templates but never
    across different templates or field-type sets (cross-template reuse would
    silently render with the wrong fields). A lookup that fails validation is
    invalidated on the spot and recomputed by the caller.
    """

    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, tuple[tuple[str, type | None], ...]], tuple[TemplatePlan, ...]
        ] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    @staticmethod
    def _key(
        template: str, field_types: dict[str, type | None]
    ) -> tuple[str, tuple[tuple[str, type | None], ...]]:
        return template, tuple(sorted(field_types.items()))

    @staticmethod
    def _parse(template: str) -> tuple[TemplatePlan, ...]:
        formatter = Formatter()
        return tuple(
            TemplatePlan(literal, field_name, format_spec, conversion)
            for literal, field_name, format_spec, conversion in formatter.parse(template)
        )

    def get(
        self, template: str, field_types: dict[str, type | None]
    ) -> tuple[TemplatePlan, ...] | None:
        """
        Return a cached plan only when it exactly matches the template text and
        allowed-field mapping. Any entry that fails this guard (e.g. a stale
        entry written for another template) is invalidated and reported as a
        miss so the caller recomputes it.
        """
        key = self._key(template, field_types)
        with self._lock:
            plans = self._entries.get(key)
            if plans is None:
                self.misses += 1
                return None
            if not _plans_match(plans, template):
                del self._entries[key]
                self.invalidations += 1
                self.misses += 1
                return None
            self.hits += 1
            return plans

    def put(
        self,
        template: str,
        field_types: dict[str, type | None],
        plans: tuple[TemplatePlan, ...],
    ) -> None:
        with self._lock:
            self._entries[self._key(template, field_types)] = plans

    def invalidate(
        self, template: str | None = None, field_types: dict[str, type | None] | None = None
    ) -> None:
        """
        Drop cached plan(s). With no arguments, clear everything; otherwise drop
        exactly the entry matching the template/field-type key.
        """
        with self._lock:
            if template is None:
                self.invalidations += len(self._entries)
                self._entries.clear()
                return
            key = self._key(template, field_types or {})
            if key in self._entries:
                del self._entries[key]
                self.invalidations += 1

    def invalidate_template_text(self, template: str) -> None:
        """Drop every cached field-type variant of one template string."""
        with self._lock:
            stale = [key for key in self._entries if key[0] == template]
            for key in stale:
                del self._entries[key]
            self.invalidations += len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _plans_match(plans: tuple[TemplatePlan, ...], template: str) -> bool:
    # Rebuild the literal template from the stored plan. A mismatch proves the
    # cached entry is stale for this template and must be recomputed.
    rebuilt: list[str] = []
    for plan in plans:
        rebuilt.append(plan.literal)
        if plan.field_name is not None:
            rebuilt.append("{")
            rebuilt.append(plan.field_name)
            if plan.conversion:
                rebuilt.append("!")
                rebuilt.append(plan.conversion)
            if plan.format_spec:
                rebuilt.append(":")
                rebuilt.append(plan.format_spec)
            rebuilt.append("}")
    return "".join(rebuilt) == template


@dataclass(frozen=True)
class StringTemplate:
    """
    A validated template string that supports only specified fields.
    Can subclass to have a type with a given set of `allowed_fields`.
    Provide a type with a field name to allow validation of int/float format strings.

    Examples:
    >>> t = StringTemplate("{name} is {age} years old", ["name", "age"])
    >>> t.format(name="Alice", age=30)
    'Alice is 30 years old'

    >>> t = StringTemplate("{count:3d}@{price:.2f}", [("count", int), ("price", float)])
    >>> t.format(count=10, price=19.99)
    ' 10@19.99'
    """

    template: str

    allowed_fields: Sequence[str | tuple[str, type | None]]
    """List of allowed field names. If `d` or `f` formats are used, give tuple with the type."""
    # Sequence is covariant so compatible with List[str]

    strict: bool = False
    """If True, raise a ValueError if the template is missing an allowed field."""

    def __post_init__(self):
        if not isinstance(self.template, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Template must be a string")

        # Confirm only the allowed fields are in the template.
        field_types = self._field_types()
        try:
            placeholder_values = {field: (type or str)(123) for field, type in field_types.items()}
            self.template.format(**placeholder_values)
        except KeyError as e:
            raise ValueError(f"Template contains unsupported variable: {e}") from None
        except ValueError as e:
            raise ValueError(
                f"Invalid template (forgot to provide a type when using non-str format strings?): {e}"
            ) from None
        # Warm the shared cache so the first format() call already benefits from
        # the parsed plan while validation above guarantees the template is safe.
        self._get_plans(field_types)

    # Shared, process-wide cache. Keying by (template text, field types) makes
    # reuse fast across identical StringTemplate instances while isolating
    # distinct templates from each other.
    cache: ClassVar[TemplateCache] = TemplateCache()

    def _field_types(self) -> dict[str, type | None]:
        return {
            field[0] if isinstance(field, tuple) else field: (
                field[1] if isinstance(field, tuple) else None
            )
            for field in self.allowed_fields
        }

    def _get_plans(self, field_types: dict[str, type | None]) -> tuple[TemplatePlan, ...]:
        plans = self.cache.get(self.template, field_types)
        if plans is None:
            plans = TemplateCache._parse(self.template)
            self.cache.put(self.template, field_types, plans)
        return plans

    @classmethod
    def invalidate_cache(cls, template: str | None = None) -> None:
        """
        Clear parsed-plan cache entries. Pass a template string to drop just that
        template (across any field-type mapping), or nothing to clear all.
        """
        if template is None:
            cls.cache.invalidate()
            return
        # The cache stores per-field-type variants of a template, so iterate the
        # entries and remove every variant of the given template text.
        cls.cache.invalidate_template_text(template)

    def format(self, **kwargs: Any) -> str:
        field_types = self._field_types()
        allowed_keys = field_types.keys()
        unexpected_keys = set(kwargs.keys()) - allowed_keys
        if self.strict and unexpected_keys:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(unexpected_keys)}")

        # Type check the values, if types were provided.
        for f, expected_type in field_types.items():
            if f in kwargs and expected_type:
                if not isinstance(kwargs[f], expected_type):
                    raise ValueError(
                        f"Invalid type for '{f}': expected {expected_type.__name__} but got {repr(kwargs[f])} ({type(kwargs[f]).__name__})"
                    )

        return self._format_cached(field_types, kwargs)

    def _format_cached(self, field_types: dict[str, type | None], kwargs: dict[str, Any]) -> str:
        def attempt(plans: tuple[TemplatePlan, ...]) -> str:
            formatter = Formatter()
            parts: list[str] = []
            for plan in plans:
                parts.append(plan.literal)
                if plan.field_name is None:
                    continue
                obj, _used = formatter.get_field(plan.field_name, (), kwargs)
                obj = formatter.convert_field(obj, plan.conversion)
                # Expand nested fields in the spec the same way str.format does;
                # a bare spec such as "3d" passes through untouched.
                spec, _ = formatter._vformat(plan.format_spec or "", (), kwargs, set(), 2)
                parts.append(formatter.format_field(obj, spec))
            return "".join(parts)

        plans = self._get_plans(field_types)
        try:
            return attempt(plans)
        except (KeyError, IndexError, AttributeError, ValueError):
            # A cached plan that does not rebuild to this template is a failed
            # hit: invalidate it and recompute once rather than rendering junk.
            if _plans_match(plans, self.template):
                raise
            self.cache.invalidate(self.template, field_types)
            return attempt(self._get_plans(field_types))

    def __bool__(self) -> bool:
        return bool(self.template)
