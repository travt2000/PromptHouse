"""Prompt templates (prompts/*.md): syntax, includes, errors, hot reload, and that every shipped file renders."""
import os
import time

import pytest

from app import prompt_files as pf
from app.style.guides import GUIDES, type_vars
from app.style.kinds import KINDS


@pytest.fixture
def pdir(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "PROMPTS_DIR", tmp_path)
    pf.clear_cache()
    yield tmp_path
    pf.clear_cache()


def write(d, name, text):
    p = d / f"{name}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_variables_sections_comments_and_standalone_lines(pdir):
    write(pdir, "t", "<!--\nfor editors\n-->\nHello {{name}}.\n{{#extra}}\nExtra: {{extra}}\n{{/extra}}\n"
                     "{{^extra}}\nNo extra.\n{{/extra}}\nEnd{{#flag}}!{{/flag}}\n\n\n")
    assert pf.render("t", name="Ann", extra="", flag=True) == "Hello Ann.\nNo extra.\nEnd!"
    assert pf.render("t", name="Ann", extra="x", flag=False) == "Hello Ann.\nExtra: x\nEnd"


def test_values_are_not_reparsed(pdir):
    write(pdir, "t", "{{a}}")
    assert pf.render("t", a="{{b}} {{#c}}") == "{{b}} {{#c}}"


def test_nested_sections_and_includes(pdir):
    write(pdir, "inner/x", "[{{#on}}in {{v}}{{/on}}]\n")
    write(pdir, "outer", "{{#v}}A{{#on}}B{{/on}}{{/v}} {{> inner/x}} done")
    assert pf.render("outer", v="1", on=True) == "AB [in 1] done"
    assert pf.render("outer", v="", on=False) == " [] done"


def test_errors(pdir):
    write(pdir, "typo", "Hi {{nmae}}")
    with pytest.raises(pf.PromptFileError, match=r"\{\{nmae\}\}.*available: name"):
        pf.render("typo", name="x")
    write(pdir, "sec", "{{#nope}}x{{/nope}}")
    with pytest.raises(pf.PromptFileError, match="nope"):
        pf.render("sec")
    write(pdir, "open", "{{#a}}never closed")
    with pytest.raises(pf.PromptFileError, match="unmatched"):
        pf.render("open", a=True)
    with pytest.raises(pf.PromptFileError, match="not found"):
        pf.render("missing")
    write(pdir, "loop1", "{{> loop2}}")
    write(pdir, "loop2", "{{> loop1}}")
    with pytest.raises(pf.PromptFileError, match="loop"):
        pf.render("loop1")
    for bad in ("../secret", "a/../../b", "C:/x", "/abs"):
        with pytest.raises(pf.PromptFileError, match="Invalid"):
            pf.render(bad)


def test_edits_take_effect_without_restart(pdir):
    p = write(pdir, "t", "one")
    assert pf.render("t") == "one"
    p.write_text("two, now longer", encoding="utf-8")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert pf.render("t") == "two, now longer"


def test_crlf_files(pdir):
    (pdir / "t.md").write_bytes(b"<!-- c -->\r\nLine 1\r\n{{#a}}\r\nLine 2\r\n{{/a}}\r\n")
    assert pf.render("t", a=True) == "Line 1\nLine 2"


# ------------------------------------------------------------------ the shipped prompt files
def test_every_shipped_prompt_renders():
    pf.clear_cache()
    for k in KINDS.values():
        assert k.system_prompt.endswith("matches the provided schema and nothing else.")
        assert "<!--" not in k.system_prompt and "{{" not in k.system_prompt
        assert k.user_prompt() in ("Assess the character in this image.", "Analyze this inspiration image.")
        assert k.user_prompt('{"x": 1}').endswith('\nJSON schema to follow:\n{"x": 1}')
    for g in GUIDES.values():
        assert g.sections[0] == "Overview" and "{{" not in g.system_prompt and "Rules:\n- Base everything" in g.system_prompt
    assert GUIDES["character"].sections[-1] == "Gaps & Uncertainties" and len(GUIDES["character"].sections) == 9
    assert GUIDES["inspiration"].sections[-1] == "Gaps" and len(GUIDES["inspiration"].sections) == 10
    assert pf.render("shared/character-type", **type_vars(None)) == "Character type: not set."
    shipped = {p.relative_to(pf.PROMPTS_DIR).with_suffix("").as_posix() for p in pf.PROMPTS_DIR.rglob("*.md")}
    assert {"analysis/character.system", "guides/user", "image-prompts/system", "image-prompts/user"} <= shipped
