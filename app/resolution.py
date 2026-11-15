"""Resolution calculator: aspect ratio + megapixel target + dimension multiple -> (width, height).

Pure Python. The frontend fetches the computed table from the backend (GET /api/resolutions) and the
backend recomputes/validates dimensions on every generate request, so the two can never disagree.

Megapixels use the 1 MP = 1024 * 1024 pixel convention, so 1:1 @ 1.0 MP is exactly 1024x1024
(Krea 2's native training resolution).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "3:4": (3, 4),
    "4:5": (4, 5),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "3:1": (3, 1),
    "21:9": (21, 9),
}
MEGAPIXELS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
MULTIPLES: tuple[int, ...] = (16, 32, 64)

DEFAULT_ASPECT = "1:1"
DEFAULT_MEGAPIXELS = 1.0
DEFAULT_MULTIPLE = 16

PIXELS_PER_MP = 1024 * 1024

# Krea 2 latent grid: VAE compression 8 x patch size 2 -> sides must be multiples of 16 (sampling.py `align`).
MODEL_ALIGNMENT = 16
# Krea 2 Turbo is documented for "1k ~ 2k resolution" (krea-2 README). Outside that range we warn.
RECOMMENDED_MIN_PIXELS = int(0.9 * PIXELS_PER_MP)
RECOMMENDED_MAX_PIXELS = 2048 * 2048
RECOMMENDED_MAX_SIDE = 2048
# Hard limits: reject instead of silently resizing. 4.0 MP presets land at ~2048^2; allow small rounding overshoot.
HARD_MAX_PIXELS = int(4.3 * PIXELS_PER_MP)
HARD_MIN_SIDE = 256
HARD_MAX_SIDE = 4096


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int
    aspect: str
    megapixels_target: float
    multiple: int
    actual_megapixels: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["warnings"] = list(self.warnings)
        return d


class ResolutionError(ValueError):
    pass


def _candidates(ideal: float, multiple: int) -> list[int]:
    base = int(ideal // multiple) * multiple
    out = {base + k * multiple for k in (-1, 0, 1, 2)}
    return sorted(v for v in out if v >= multiple)


def compute_dimensions(aspect: str, megapixels: float, multiple: int) -> tuple[int, int]:
    """Pick (w, h), both multiples of `multiple`, minimizing aspect-ratio and area deviation."""
    if aspect not in ASPECT_RATIOS:
        raise ResolutionError(f"Unsupported aspect ratio {aspect!r}; choose one of {list(ASPECT_RATIOS)}")
    if multiple not in MULTIPLES:
        raise ResolutionError(f"Unsupported dimension multiple {multiple}; choose one of {list(MULTIPLES)}")
    if megapixels not in MEGAPIXELS:
        raise ResolutionError(f"Unsupported megapixel preset {megapixels}; choose one of {list(MEGAPIXELS)}")

    aw, ah = ASPECT_RATIOS[aspect]
    ratio = aw / ah
    target = megapixels * PIXELS_PER_MP
    ideal_w = math.sqrt(target * ratio)
    ideal_h = math.sqrt(target / ratio)

    best: tuple[float, int, int] | None = None
    for w in _candidates(ideal_w, multiple):
        for h in _candidates(ideal_h, multiple):
            aspect_err = math.log((w / h) / ratio)
            area_err = math.log((w * h) / target)
            score = aspect_err**2 + area_err**2
            key = (score, -(w * h), w)  # deterministic tie-break: prefer larger area, then larger width
            if best is None or key < (best[0], -(best[1] * best[2]), best[1]):
                best = (score, w, h)
    assert best is not None
    return best[1], best[2]


def validate_dimensions(width: int, height: int) -> list[str]:
    """Raise ResolutionError for unsupported sizes; return non-fatal warnings."""
    if width % MODEL_ALIGNMENT or height % MODEL_ALIGNMENT:
        raise ResolutionError(f"{width}x{height}: width and height must be multiples of {MODEL_ALIGNMENT}")
    if min(width, height) < HARD_MIN_SIDE:
        raise ResolutionError(f"{width}x{height}: sides below {HARD_MIN_SIDE}px are not supported")
    if max(width, height) > HARD_MAX_SIDE:
        raise ResolutionError(f"{width}x{height}: sides above {HARD_MAX_SIDE}px are not supported")
    if width * height > HARD_MAX_PIXELS:
        raise ResolutionError(
            f"{width}x{height} ({width * height / PIXELS_PER_MP:.2f} MP) exceeds the {HARD_MAX_PIXELS / PIXELS_PER_MP:.1f} MP limit"
        )
    warnings = []
    px = width * height
    if px < RECOMMENDED_MIN_PIXELS:
        warnings.append("Below Krea 2 Turbo's documented 1K-2K range; quality may degrade.")
    if px > RECOMMENDED_MAX_PIXELS or max(width, height) > RECOMMENDED_MAX_SIDE:
        warnings.append("Beyond 2048px / 2048x2048 is outside the documented 1K-2K range; quality may degrade.")
    return warnings


def resolve(aspect: str, megapixels: float, multiple: int) -> Resolution:
    w, h = compute_dimensions(aspect, megapixels, multiple)
    warnings = validate_dimensions(w, h)
    return Resolution(
        width=w,
        height=h,
        aspect=aspect,
        megapixels_target=megapixels,
        multiple=multiple,
        actual_megapixels=round(w * h / PIXELS_PER_MP, 3),
        warnings=tuple(warnings),
    )


def table() -> dict:
    """Every preset combination, precomputed for the frontend."""
    return {
        "aspects": list(ASPECT_RATIOS),
        "megapixels": list(MEGAPIXELS),
        "multiples": list(MULTIPLES),
        "defaults": {"aspect": DEFAULT_ASPECT, "megapixels": DEFAULT_MEGAPIXELS, "multiple": DEFAULT_MULTIPLE},
        "entries": {
            # mp formatted with :g so keys match JavaScript's Number->string ("1", "0.5", "1.5")
            f"{a}|{mp:g}|{m}": resolve(a, mp, m).to_dict() for a in ASPECT_RATIOS for mp in MEGAPIXELS for m in MULTIPLES
        },
    }
