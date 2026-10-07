"""Reproducible collision study for truncated base36 hash IDs.

Short IDs are the first ``length`` characters of the base36 encoding of a SHA1
digest (the same scheme used by ``strif.string_replace.short_id``). For each
candidate ID length this measures:

- how many inputs share a bucket (the birthday setting),
- empirical collision odds vs. the approximation
  ``1 - exp(-N * (N - 1) / (2 * 36**length))``,
- how many salted retries (the ``unique_short_id`` fallback) are needed as the
  ID table fills up.

Usage:
    python devtools/id_collision_study.py [--seed 20260929] [--results PATH]

The run is deterministic for a fixed seed and writes a JSON summary so the
report page can embed the exact numbers behind the length / retry decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
LENGTHS = (4, 5, 6, 7, 8, 9, 10, 12)
POPULATION_SIZES = (100, 1_000, 10_000, 100_000)
DEFAULT_SEED = 20260929


def base36_from_sha1(text: str, salt: int = 0) -> str:
    payload = text if salt == 0 else f"{salt}\x00{text}"
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    number = int.from_bytes(digest, byteorder="big")
    chars: list[str] = []
    while number > 0:
        number, remainder = divmod(number, 36)
        chars.append(BASE36[remainder])
    return "".join(reversed(chars))


def short_id(text: str, length: int, attempt: int = 0) -> str:
    return base36_from_sha1(text, salt=attempt)[:length].rjust(length, "0")


def make_inputs(size: int, rng: random.Random) -> list[str]:
    inputs = [f"item-{i}" for i in range(size // 2)]
    inputs.extend(
        "".join(rng.choice(BASE36) for _ in range(rng.randint(4, 24)))
        for _ in range(size - len(inputs))
    )
    rng.shuffle(inputs)
    return inputs


def birthday_probability(n: int, space: int) -> float:
    if n * (n - 1) / 2 > 700 * space:
        return 1.0
    return -math.expm1(-n * (n - 1) / (2 * space))


def measure_collisions(inputs: list[str], length: int) -> dict[str, float | int]:
    buckets: dict[str, int] = {}
    for value in inputs:
        bucket = short_id(value, length)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    occupied = len(buckets)
    colliding_buckets = sum(1 for count in buckets.values() if count > 1)
    colliding_inputs = sum(count for count in buckets.values() if count > 1)
    space = 36**length
    return {
        "length": length,
        "space": space,
        "population": len(inputs),
        "occupied_buckets": occupied,
        "colliding_buckets": colliding_buckets,
        "colliding_inputs": colliding_inputs,
        "max_multiplicity": max(buckets.values()),
        "theoretical_collision_prob": round(birthday_probability(len(inputs), space), 12),
    }


def measure_retries(inputs: list[str], length: int, cap: int = 128) -> dict[str, int]:
    used: set[str] = set()
    total_retries = 0
    exhausted = 0
    for value in inputs:
        for attempt in range(cap + 1):
            candidate = short_id(value, length, attempt)
            if candidate not in used:
                used.add(candidate)
                total_retries += attempt
                break
        else:
            exhausted += 1
    return {
        "length": length,
        "total_retries": total_retries,
        "exhausted_inputs": exhausted,
    }


def run_study(seed: int = DEFAULT_SEED) -> dict[str, object]:
    rng = random.Random(seed)
    populations = {size: make_inputs(size, rng) for size in POPULATION_SIZES}
    table: list[dict[str, float | int]] = []
    for inputs in populations.values():
        for length in LENGTHS:
            table.append(measure_collisions(inputs, length))
    retry_table = [measure_retries(populations[10_000], length) for length in LENGTHS]
    return {"seed": seed, "collisions": table, "retries": retry_table}


def format_report(results: dict[str, object]) -> str:
    lines = [
        f"seed: {results['seed']}",
        "",
        "Birthday collisions by population / ID length:",
        f"{'pop':>8} {'len':>4} {'space':>14} {'occupied':>10} "
        f"{'col buckets':>12} {'col inputs':>11} {'p(collision)':>13}",
    ]
    for row in results["collisions"]:
        lines.append(
            f"{row['population']:>8} {row['length']:>4} {row['space']:>14,} "
            f"{row['occupied_buckets']:>10,} {row['colliding_buckets']:>12,} "
            f"{row['colliding_inputs']:>11,} {row['theoretical_collision_prob']:>13.3g}"
        )
    lines += [
        "",
        "Salted retries needed over 10,000 distinct inputs:",
        f"{'len':>4} {'retries':>9} {'exhausted':>10}",
    ]
    for row in results["retries"]:
        lines.append(f"{row['length']:>4} {row['total_retries']:>9} {row['exhausted_inputs']:>10}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "id_collision_results.json",
    )
    args = parser.parse_args()

    results = run_study(args.seed)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(format_report(results))
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
