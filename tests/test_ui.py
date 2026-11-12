from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
TEXT = ("We offer a software engineering internship with a mentor. "
        "You will use Python and write tests, following a technical interview.")


def test_ui_manual_check_and_invalid_submission():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    app.text_area[0].input(TEXT)
    app.selectbox[0].select("No")
    app.selectbox[1].select("Yes")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["assessment"]["agent"]["check_id"] == "verify_company"
    app.selectbox[2].select("Yes")
    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["assessment"]["agent"]["check_id"] == "confirm_recruiter"
    app.selectbox[2].select("Unable to determine")
    app.button[1].click().run(timeout=30)
    assert app.session_state["assessment"]["agent"]["check_id"] is None
    assert app.session_state["assessment"]["evidence_completeness"] == .75
    app.text_area[0].input("short")
    app.button[0].click().run(timeout=30)
    assert app.error
    assert "assessment" not in app.session_state
