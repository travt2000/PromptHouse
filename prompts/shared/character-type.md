<!--
The character's type, from config/character_types.json. Included by guides/user.md and image-prompts/user.md.
Variables: type_label (empty when no type is set), type_description, type_guidance (either may be empty).
-->
{{#type_label}}
Character type: {{type_label}}.{{#type_description}} {{type_description}}{{/type_description}}
{{#type_guidance}}
Guidance for this type: {{type_guidance}}
{{/type_guidance}}
{{/type_label}}
{{^type_label}}
Character type: not set.
{{/type_label}}
