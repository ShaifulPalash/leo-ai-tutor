"""Headless smoke test of the Streamlit app (no browser, no API calls)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.ui
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def find_button(app, text):
    for button in app.button:
        if text in button.label:
            return button
    raise AssertionError(f"no button containing {text!r}; found {[b.label for b in app.button]}")


def test_app_renders_without_errors():
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    assert any("Agent Activity" in markdown.value for markdown in app.markdown)


def test_offline_session_teaches_a_topic():
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.toggle(key="offline_toggle").set_value(True).run()
    find_button(app, "Start / reset").click().run()
    assert not app.exception
    app.button(key="sample_Fractions").click().run()
    assert not app.exception
    assert app.session_state["session"].stage == "lesson"
