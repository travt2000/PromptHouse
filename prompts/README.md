# Prompts

Everything PromptHouse sends to the local language model (llama-server, see Settings → Style analysis) is in
this folder. Edit a file and the change applies to the **next request**; you don't need to restart the app. Each file
starts with a `<!-- ... -->` comment saying what it is for and which variables it can use. Comments are never
sent to the model.

| File | Used for |
|---|---|
| `analysis/character.system.md`, `analysis/character.user.md` | reading one image of the character (Style → Character images) |
| `analysis/inspiration.system.md`, `analysis/inspiration.user.md` | reading one inspiration image (Style → Inspiration) |
| `analysis/json-rule.md`, `analysis/schema-in-prompt.md` | shared parts of the analysis prompts |
| `guides/character.system.md` | writing the Self-Style Guide; its `## ` headings **are** the guide's sections |
| `guides/inspiration.system.md` | writing the Content Guide; likewise |
| `guides/rules.md`, `guides/user.md` | shared guide rules; the data message |
| `image-prompts/system.md`, `image-prompts/user.md` | **Write prompts** in the generator |
| `shared/character-type.md` | the character-type paragraph (types themselves: `config/character_types.json`) |

## Template syntax

| Write | Gives |
|---|---|
| `{{name}}` | the value of a variable |
| `{{#name}}text{{/name}}` | `text` only if the variable is non-empty / true |
| `{{^name}}text{{/name}}` | `text` only if the variable is empty / false |
| `{{> folder/file}}` | the contents of another file here (without `.md`), with the same variables |
| `<!-- note -->` | nothing (a note for editors) |

A line that holds nothing but a `{{#…}}`, `{{^…}}` or `{{/…}}` tag disappears completely, so sections can be
written on their own lines. Trailing blank lines are ignored. Using a variable that a file does not receive is
reported as an error naming the file, instead of sending a broken prompt.

## Keep in mind

* The analysis replies are validated against the JSON schemas in `app/style/schema.py`. The prompts can change
  how fields are filled in, but not which fields exist.
* Changing the analysis prompts does not re-analyze existing images. Use **Re-analyze all** in the Style window
  to apply a change to images that were already analyzed.
* **Write prompts** expects a numbered (or bulleted) list in the reply, one prompt per item.
* `PROMPTHOUSE_PROMPTS_DIR` makes the app use a different folder, for example to try out a variant.
