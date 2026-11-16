import copy
from datetime import datetime

from app.style.aggregate import aggregate, prompt_summary, stats_markdown, tier
from app.style.character_types import CharacterType
from app.style.vsettings import VisionSettings as Settings
from app.style.guides import GUIDES, build_messages, clean_body, compose_document, prompt_budget_chars
from app.style.kinds import CHARACTER, INSPIRATION
from app.style.schema import sample_record


def field(stats, path):
    return next(f for f in stats["fields"] if f["path"] == path)


def insp(shot="half_body", themes=("fashion",), garments=(), notes="soft window light"):
    rec = copy.deepcopy(sample_record(INSPIRATION.schema))
    rec["photography"]["shot_type"] = shot
    rec["photography"]["notes"] = notes
    rec["content"]["themes"] = list(themes)
    rec["outfit"]["garments"] = [{"category": c, "description": d, "color": col} for c, d, col in garments]
    return rec


def test_enum_shares_ignore_unknown_and_report_coverage():
    stats = aggregate(INSPIRATION, [insp("half_body"), insp("half_body"), insp("full_body"), insp("unknown")])
    f = field(stats, "photography.shot_type")
    assert f["values"] == [["half_body", 0.667], ["full_body", 0.333]]
    assert f["coverage"] == 0.75
    assert stats["image_count"] == 4 and not stats["weighted"]


def test_lists_count_images_not_mentions():
    stats = aggregate(INSPIRATION, [insp(themes=("fashion", "travel")), insp(themes=("fashion",)), insp(themes=())])
    assert dict(map(tuple, field(stats, "content.themes")["values"])) == {"fashion": 0.667, "travel": 0.333}


def test_garments_split_into_categories_and_descriptions():
    stats = aggregate(INSPIRATION, [
        insp(garments=[("top", "cropped knit", "cream"), ("bottom", "wide-leg jeans", "blue")]),
        insp(garments=[("top", "Cropped knit", "Cream"), ("top", "tank", "")]),
    ])
    assert dict(map(tuple, field(stats, "outfit.garments.category")["values"])) == {"top": 1.0, "bottom": 0.5}
    items = field(stats, "outfit.garments")["items"]
    assert items[0] == ["cropped knit cream", 2]  # case-insensitive, first spelling kept
    assert ["tank", 1] in items


def test_text_is_deduplicated_and_blanks_dropped():
    stats = aggregate(INSPIRATION, [insp(notes="Soft  light"), insp(notes="soft light"), insp(notes="unknown"), insp(notes="")])
    assert field(stats, "photography.notes")["items"] == [["Soft light", 2]]


def test_character_records_are_weighted_by_confidence():
    def char(hair, conf):
        rec = copy.deepcopy(sample_record(CHARACTER.schema))
        rec["appearance"]["hair_color"] = hair
        rec["image_context"]["assessment_confidence"] = conf
        return rec
    stats = aggregate(CHARACTER, [char("auburn", "high"), char("auburn", "high"), char("black", "low")])
    values = dict(map(tuple, field(stats, "appearance.hair_color")["values"]))
    assert values["auburn"] > 0.85 and values["black"] < 0.15  # 2.0 vs 0.3
    assert stats["weighted"]


def test_tiers_and_rendering():
    assert (tier(0.6), tier(0.3), tier(0.1)) == ("signature", "common", "occasional")
    stats = aggregate(INSPIRATION, [insp("half_body")] * 3 + [insp("full_body")])
    summary = prompt_summary(stats, text_items=3)
    assert "shot type: half body 75% [signature], full body 25% [common]" in summary
    assert "[photography]" in summary
    assert "soft window light (x4)" in summary
    assert "soft window light" not in prompt_summary(stats, text_items=0)
    appendix = stats_markdown(stats)
    assert appendix.startswith("## Appendix: Data Summary") and "| shot type | half body 75% [signature]" in appendix


def test_messages_include_character_type_and_fit_budget():
    stats = aggregate(INSPIRATION, [insp(notes="x" * 250 + str(i)) for i in range(40)])
    ctype = CharacterType("cosplay", "Cosplay Character", "Costumes.", "Treat the look as a canvas.")
    roomy = build_messages(INSPIRATION, stats, ctype, budget_chars=100_000)
    tight = build_messages(INSPIRATION, stats, ctype, budget_chars=4_500)
    assert roomy[0]["content"] == GUIDES["inspiration"].system_prompt
    user = tight[1]["content"]
    assert "Character type: Cosplay Character. Costumes.\nGuidance for this type: Treat the look as a canvas." in user
    assert len(tight[0]["content"]) + len(user) <= 4_500 < len(roomy[0]["content"]) + len(roomy[1]["content"])
    assert "Character type: not set." in build_messages(INSPIRATION, stats, None, 100_000)[1]["content"]
    assert prompt_budget_chars(Settings(n_ctx=8192, guide_max_tokens=3072)) == (8192 - 3072 - 512) * 3


def test_inspiration_guide_prompt_forbids_physical_traits():
    assert "never describe the physical traits" in GUIDES["inspiration"].system_prompt
    assert "Prompt Keywords" in GUIDES["character"].sections
    assert "Content Ideas" in GUIDES["inspiration"].sections


def test_clean_body_strips_think_fences_and_title():
    assert clean_body("<think>x</think>```markdown\n# Title\n\n## Overview\nHi\n```") == "## Overview\nHi"


def test_compose_document_adds_header_notes_and_appendix():
    stats = aggregate(CHARACTER, [sample_record(CHARACTER.schema)])
    body = "## Overview\nA.\n\n## Colour Palette\nB."
    doc = compose_document(CHARACTER, body, stats, "model.gguf", None, datetime(2026, 11, 15, 9, 30), truncated=True)
    assert doc.startswith("# Self-Style Guide\n\n*Generated 2026-11-15 09:30 · 1 images · model model.gguf")
    assert "character type: not set" in doc and "output limit" in doc
    assert "left out these sections: Defining Features" in doc and "Colour Palette," not in doc.split("left out")[1].split("\n")[0]
    assert doc.rstrip().endswith("|") and "## Appendix: Data Summary" in doc
