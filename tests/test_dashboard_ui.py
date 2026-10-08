"""Optional-dependency checks for reproducible notebooks and applied dashboard state."""
# TEST LOGIC: These tests use synthetic inputs and never save executed notebook outputs.
import ast
from pathlib import Path
import pytest


# TEST LOGIC: Resolve artifacts relative to this repository rather than the caller's current directory.
PROJECT = Path(__file__).resolve().parents[1]


def _paired_cells(source):
    # TEST LOGIC: Read percent-format boundaries and normalize only their documented Markdown prefix.
    cells, lines, kind = [], [], None
    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "".join(lines).strip("\n")))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line)
        else:
            lines.append(line)
    if kind is not None:
        cells.append((kind, "".join(lines).strip("\n")))
    return cells


def test_notebook_matches_paired_source_and_has_no_saved_outputs():
    # TEST LOGIC: Exact source equality includes executable comments and examples, preventing documentation drift.
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(PROJECT / "jump_filter_dashboard.ipynb", as_version=4)
    source = (PROJECT / "examples" / "dashboard.py").read_text(encoding="utf-8")
    nbformat.validate(notebook)
    assert [(cell.cell_type, cell.source) for cell in notebook.cells] == _paired_cells(source)
    assert len([cell for cell in notebook.cells if cell.cell_type == "code"]) >= 2
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == []
            assert cell.execution_count is None


def test_streamlit_apply_controls_and_failed_apply_preserve_review():
    # TEST LOGIC: Skip cleanly for a minimal engine install; browser tests run when its optional extra is present.
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    # TEST LOGIC: A successful Apply creates charts, explicit statistical tables and original-order annotations.
    app = AppTest.from_file(str(PROJECT / "jump_filter" / "app.py"), default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    review = app.session_state["jf_review"]
    assert len(review["result"]) == len(review["data"])
    assert review["result"].index.equals(review["data"].index)
    assert len(app.metric) == 6
    assert len(app.get("plotly_chart")) >= 2
    assert len(app.dataframe) >= 4

    # TEST LOGIC: CUSIP browsing changes the view while method edits remain pending until Apply.
    focus = next(box for box in app.selectbox if box.label == "Bond / CUSIP")
    focus.set_value(focus.options[1]).run()
    assert not app.exception
    assert app.session_state["jf_review"] is review
    next(box for box in app.selectbox if box.label == "Method").set_value("hampel").run()
    assert app.session_state["jf_review"]["config"].method == "consensus"
    assert any("Pending changes" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    assert app.session_state["jf_review"]["config"].method == "hampel"

    # TEST LOGIC: Both edit directions update the synchronized slider and exact numerical input.
    app.slider(key="jf_slider_threshold").set_value(5.2).run()
    assert app.number_input(key="jf_number_threshold").value == 5.2
    app.number_input(key="jf_number_threshold").set_value(4.9).run()
    assert app.slider(key="jf_slider_threshold").value == 4.9

    # TEST LOGIC: An incompatible reference count preserves the valid snapshot rather than publishing partial state.
    previous = app.session_state["jf_review"]
    app.number_input(key="jf_number_min_neighbors").set_value(60).run()
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    assert app.session_state["jf_review"] is previous
    assert any("Could not apply" in error.value for error in app.error)

    # TEST LOGIC: Explicit comparison uses the applied configuration and every supported method on one fixed bond.
    next(button for button in app.button if button.label == "Compare methods for this bond").click().run()
    assert not app.exception
    comparison = app.session_state["jf_comparison"][1]
    from jump_filter import METHODS
    assert list(comparison["method"]) == list(METHODS)
    assert comparison["total"].nunique() == 1
    assert app.session_state["jf_review"] is previous
