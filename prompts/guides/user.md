<!--
The user message for both guides: the character type, then the tallied data.
Variables:
  title         "Self-Style Guide" or "Content Guide"
  image_count   number of analysed images
  weighted      true for character images (shares weighted by the model's confidence)
  data          the tally, formatted by code (app/style/aggregate.py); trimmed to fit the context size
  type_label, type_description, type_guidance   the character type (see shared/character-type.md)
-->
{{> shared/character-type}}

Data from {{image_count}} analyzed images.{{#weighted}} Shares are weighted by how reliable each image's assessment was.{{/weighted}}

{{data}}

Write the {{title}}.
