"""The extraction schemas (versioned, one per analysis kind).

Controlled-vocabulary fields are enums so records can be aggregated later ("73% of the
references are golden-hour three-quarter shots"). Free-text fields hold nuance the enums
miss. Every enum includes ``unknown`` so the model is never forced to guess.

Two schemas share one vocabulary:

* **character**: an assessment of the creator's own character (hair, colouring, build, what
  flatters them) used to recommend a personal "self style".
* **inspiration**: how a piece of content was made and styled (format, camera, light, pose,
  outfit, setting, mood). It deliberately has no fields for the subject's physical traits.

Bump a kind's schema version whenever one of its fields or enum values changes; analyses are
stored with the version and model that produced them, so stale ones can be re-run selectively.
"""
from __future__ import annotations

CHARACTER_SCHEMA_VERSION = 1
INSPIRATION_SCHEMA_VERSION = 2  # v1 was the original single-purpose schema
UNKNOWN = "unknown"

VOCAB: dict[str, list[str]] = {
    # --- photography ---
    "shot_type": ["extreme_close_up", "close_up", "headshot", "half_body", "three_quarter", "full_body", "wide_environmental", "detail"],
    "camera_angle": ["eye_level", "low_angle", "high_angle", "overhead", "dutch", "over_the_shoulder"],
    "framing": ["centered", "rule_of_thirds", "off_center", "symmetrical", "tight_crop", "negative_space"],
    "depth_of_field": ["shallow", "medium", "deep"],
    "lens_feel": ["wide", "normal", "short_telephoto", "telephoto", "macro", "fisheye"],
    "lighting_type": ["natural_window", "golden_hour", "overcast", "direct_sun", "studio_softbox", "hard_flash", "on_camera_flash", "neon", "low_key_candlelight", "mixed_ambient", "backlit"],
    "lighting_direction": ["front", "side", "back", "top", "under", "rim", "mixed"],
    "color_grade": ["neutral", "warm", "cool", "desaturated", "vibrant", "faded_film", "high_contrast", "moody_dark", "pastel", "monochrome"],
    "medium_look": ["digital_clean", "film_grain", "polaroid", "phone_snapshot", "editorial_glossy", "cinematic", "vintage"],
    # --- pose ---
    "body_orientation": ["facing_camera", "three_quarter_turn", "profile", "back_to_camera", "over_shoulder"],
    "posture": ["standing", "sitting", "leaning", "lying", "kneeling", "crouching", "walking", "dancing", "reclining"],
    "head_tilt": ["neutral", "tilted", "down", "up", "turned"],
    "gaze": ["at_camera", "away", "down", "up", "closed_eyes", "off_frame"],
    "expression": ["neutral", "soft_smile", "big_smile", "laughing", "serious", "sultry", "playful", "pensive", "surprised"],
    "pose_energy": ["candid", "posed", "editorial", "dynamic", "relaxed"],
    # --- outfit ---
    "garment_category": ["top", "bottom", "dress", "outerwear", "swimwear", "lingerie", "activewear", "footwear", "headwear", "legwear", "one_piece", "other"],
    "outfit_style": ["casual", "streetwear", "athleisure", "formal", "boho", "vintage", "minimalist", "glam", "sporty", "cozy", "preppy", "edgy", "romantic", "other"],
    # --- setting / mood ---
    "location_type": ["home_interior", "studio", "cafe_restaurant", "urban_street", "nature", "beach", "pool", "vehicle", "nightlife", "gym", "hotel", "shop", "other_indoor", "other_outdoor", "plain_backdrop"],
    "time_of_day": ["dawn", "morning", "midday", "golden_hour", "dusk", "night", "indeterminate"],
    "vibe": ["cozy", "glamorous", "playful", "intimate", "moody", "energetic", "serene", "edgy", "nostalgic", "confident", "candid_everyday", "luxurious"],
    # --- content (inspiration) ---
    "content_format": ["selfie", "mirror_selfie", "portrait", "lifestyle_candid", "outfit_showcase", "editorial", "close_up_detail", "behind_the_scenes", "product_feature", "flat_lay", "scenery"],
    "content_theme": ["fashion", "beauty", "lifestyle", "travel", "fitness", "food_drink", "home", "nightlife", "wellness", "seasonal_holiday", "romance", "work_study", "art_creative"],
    "subject_count": ["none", "solo", "duo", "group"],
    # --- appearance (character) ---
    "hair_color": ["black", "dark_brown", "brown", "light_brown", "auburn", "red", "copper", "strawberry_blonde", "dark_blonde", "blonde", "platinum", "gray_silver", "white", "pastel_dyed", "vivid_dyed", "multi_tone"],
    "hair_length": ["very_short", "short", "chin_length", "shoulder_length", "mid_back", "waist_length"],
    "hair_texture": ["straight", "wavy", "curly", "coily"],
    "eye_color": ["dark_brown", "brown", "hazel", "amber", "green", "blue", "gray"],
    "skin_tone": ["very_fair", "fair", "light", "light_medium", "medium", "tan", "deep", "very_deep"],
    "skin_undertone": ["warm", "cool", "neutral", "olive"],
    "face_shape": ["oval", "round", "square", "heart", "oblong", "diamond", "triangle"],
    "build": ["petite", "slender", "athletic", "average", "curvy", "full_figured", "muscular"],
    "contrast_level": ["low", "medium", "high"],
    "color_season": ["spring", "summer", "autumn", "winter"],
    "confidence": ["low", "medium", "high"],
}


def _enum(name: str) -> dict:
    return {"type": "string", "enum": VOCAB[name] + [UNKNOWN]}


def _enum_list(name: str, max_items: int) -> dict:
    return _list_of({"type": "string", "enum": VOCAB[name]}, max_items)


def _text(max_len: int = 300) -> dict:
    return {"type": "string", "maxLength": max_len}


def _list_of(item: dict, max_items: int) -> dict:
    return {"type": "array", "items": item, "maxItems": max_items}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _garments() -> dict:
    return _list_of(_obj({
        "category": _enum("garment_category"),
        "description": _text(120),
        "color": _text(40),
    }), 8)


def build_inspiration_schema() -> dict:
    return _obj({
        "content": _obj({
            "format": _enum("content_format"),
            "themes": _enum_list("content_theme", 3),
            "subject_count": _enum("subject_count"),
            "concept": _text(200),
            "takeaways": _list_of(_text(160), 4),
        }),
        "photography": _obj({
            "shot_type": _enum("shot_type"),
            "camera_angle": _enum("camera_angle"),
            "framing": _enum("framing"),
            "depth_of_field": _enum("depth_of_field"),
            "lens_feel": _enum("lens_feel"),
            "lighting_type": _enum("lighting_type"),
            "lighting_direction": _enum("lighting_direction"),
            "color_grade": _enum("color_grade"),
            "dominant_colors": _list_of(_text(30), 5),
            "medium_look": _enum("medium_look"),
            "notes": _text(),
        }),
        "pose": _obj({
            "body_orientation": _enum("body_orientation"),
            "posture": _enum("posture"),
            "head_tilt": _enum("head_tilt"),
            "gaze": _enum("gaze"),
            "expression": _enum("expression"),
            "energy": _enum("pose_energy"),
            "hands_and_arms": _text(),
            "description": _text(),
        }),
        "outfit": _obj({
            "style": _enum("outfit_style"),
            "garments": _garments(),
            "accessories": _list_of(_text(60), 8),
            "hair_styling": _text(120),
            "makeup": _text(120),
        }),
        "setting": _obj({
            "location_type": _enum("location_type"),
            "time_of_day": _enum("time_of_day"),
            "props": _list_of(_text(60), 8),
            "description": _text(),
        }),
        "mood": _obj({
            "vibe": _enum_list("vibe", 3),
            "essence": _text(400),
        }),
    })


def build_character_schema() -> dict:
    return _obj({
        "image_context": _obj({
            "shot_type": _enum("shot_type"),
            "lighting_type": _enum("lighting_type"),
            "color_grade": _enum("color_grade"),
            "assessment_confidence": _enum("confidence"),
            "confidence_notes": _text(200),
        }),
        "appearance": _obj({
            "hair_color": _enum("hair_color"),
            "hair_length": _enum("hair_length"),
            "hair_texture": _enum("hair_texture"),
            "hair_styling": _text(120),
            "eye_color": _enum("eye_color"),
            "skin_tone": _enum("skin_tone"),
            "skin_undertone": _enum("skin_undertone"),
            "face_shape": _enum("face_shape"),
            "build": _enum("build"),
            "distinctive_features": _list_of(_text(60), 6),
        }),
        "color_analysis": _obj({
            "contrast_level": _enum("contrast_level"),
            "color_season": _enum("color_season"),
            "flattering_colors": _list_of(_text(30), 6),
            "colors_to_avoid": _list_of(_text(30), 4),
            "notes": _text(),
        }),
        "current_look": _obj({
            "style": _enum("outfit_style"),
            "garments": _garments(),
            "accessories": _list_of(_text(60), 8),
            "makeup": _text(120),
            "works_well": _text(),
            "works_less_well": _text(),
        }),
        "on_camera": _obj({
            "expression": _enum("expression"),
            "gaze": _enum("gaze"),
            "energy": _enum("pose_energy"),
            "angle_notes": _text(200),
        }),
        "recommendations": _obj({
            "style_directions": _enum_list("outfit_style", 3),
            "silhouettes": _list_of(_text(80), 4),
            "hair_and_makeup": _text(),
            "summary": _text(400),
        }),
    })


INSPIRATION_SCHEMA = build_inspiration_schema()
CHARACTER_SCHEMA = build_character_schema()


def sample_record(schema: dict) -> dict:
    """A valid record for ``schema`` (first enum value everywhere). Used by tests and the fake server."""
    def fill(node):
        t = node.get("type")
        if t == "object":
            return {k: fill(v) for k, v in node["properties"].items()}
        if t == "array":
            item = node["items"]
            return [fill(item)] if item.get("type") == "object" else ([item["enum"][0]] if "enum" in item else ["sample"])
        if "enum" in node:
            return node["enum"][0]
        return "sample"
    return fill(schema)
