<!--
The user message for "Write prompts".
Variables:
  self_style_guide    the Self-Style Guide (prompt-keyword sections first; appendix and Gaps removed), or empty
  content_guide       the Content Guide, likewise, or empty
  has_content_guide   true if there is a Content Guide
  idea                the creator's optional direction ("rainy city night"), or empty
  count               how many prompts to write
  plural              true if count is not 1
  type_label, type_description, type_guidance   the character type (see shared/character-type.md)
The guides are trimmed to fit the context size; everything outside them is never trimmed.
-->
{{#self_style_guide}}
=== Self-Style Guide ===
{{self_style_guide}}

{{/self_style_guide}}
{{#content_guide}}
=== Content Guide ===
{{content_guide}}

{{/content_guide}}
{{> shared/character-type}}

{{#idea}}Direction from the creator (follow it, within the guides): {{idea}}{{/idea}}{{^idea}}No specific direction: pick varied ideas from the Content Guide's pillars and content ideas.{{/idea}}{{^has_content_guide}} There is no Content Guide, so choose settings and formats that suit the character.{{/has_content_guide}}

Write {{count}} prompt{{#plural}}s{{/plural}}.
