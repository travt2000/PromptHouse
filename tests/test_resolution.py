import itertools
import math

import pytest

from app import resolution as R


def test_defaults_are_1024_square():
    r = R.resolve(R.DEFAULT_ASPECT, R.DEFAULT_MEGAPIXELS, R.DEFAULT_MULTIPLE)
    assert (r.width, r.height) == (1024, 1024)
    assert r.actual_megapixels == 1.0
    assert r.warnings == ()


@pytest.mark.parametrize("aspect,mp,mult", list(itertools.product(R.ASPECT_RATIOS, R.MEGAPIXELS, R.MULTIPLES)))
def test_every_preset_is_valid_and_close(aspect, mp, mult):
    r = R.resolve(aspect, mp, mult)
    assert r.width % mult == 0 and r.height % mult == 0
    assert r.width % 16 == 0 and r.height % 16 == 0  # Krea 2 latent/patch alignment
    aw, ah = R.ASPECT_RATIOS[aspect]
    # within one multiple step of the ideal aspect / area
    assert abs(math.log((r.width / r.height) / (aw / ah))) < 0.12
    assert abs(math.log(r.width * r.height / (mp * R.PIXELS_PER_MP))) < 0.15


def test_known_values():
    assert (lambda r: (r.width, r.height))(R.resolve("16:9", 1.0, 16)) == (1360, 768)
    assert (lambda r: (r.width, r.height))(R.resolve("1:1", 4.0, 64)) == (2048, 2048)
    assert (lambda r: (r.width, r.height))(R.resolve("9:16", 1.0, 64)) == (768, 1344)


def test_orientation_follows_aspect():
    assert R.resolve("3:4", 1.0, 16).height > R.resolve("3:4", 1.0, 16).width
    assert R.resolve("21:9", 1.0, 16).width > R.resolve("21:9", 1.0, 16).height


def test_warnings_outside_documented_range():
    assert R.resolve("1:1", 0.5, 16).warnings  # below 1K
    assert R.resolve("3:1", 4.0, 16).warnings  # long side > 2048


def test_rejects_unknown_options():
    with pytest.raises(R.ResolutionError):
        R.resolve("2:1", 1.0, 16)
    with pytest.raises(R.ResolutionError):
        R.resolve("1:1", 1.25, 16)
    with pytest.raises(R.ResolutionError):
        R.resolve("1:1", 1.0, 8)


def test_validate_dimensions_rejects_bad_sizes():
    with pytest.raises(R.ResolutionError):
        R.validate_dimensions(1000, 1024)  # not a multiple of 16
    with pytest.raises(R.ResolutionError):
        R.validate_dimensions(4096, 2048)  # too many pixels
    with pytest.raises(R.ResolutionError):
        R.validate_dimensions(128, 1024)


def test_table_keys_match_javascript_number_format():
    t = R.table()
    assert "1:1|1|16" in t["entries"] and "1:1|0.5|16" in t["entries"] and "16:9|1.5|64" in t["entries"]
    assert len(t["entries"]) == len(R.ASPECT_RATIOS) * len(R.MEGAPIXELS) * len(R.MULTIPLES)
