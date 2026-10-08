"""Execute the shipped dashboard through a real Jupyter kernel and widget protocol."""

# TEST LOGIC: Kernel execution is optional for numerical-only package installations.
import json
from pathlib import Path
import sys
import pytest

# TEST LOGIC: Resolve the shipped notebook independently of the caller's directory.
PROJECT = Path(__file__).resolve().parents[1]
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"


def _kernel_manager(directory):
    # TEST LOGIC: Select the current interpreter explicitly instead of an unrelated global python3 kernel.
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir = directory / "kernels" / "jump-filter-validation"
    kernel_dir.mkdir(parents=True)
    specification = dict(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                         display_name="Jump Filter validation", language="python")
    (kernel_dir / "kernel.json").write_text(json.dumps(specification), encoding="utf-8")
    manager = KernelSpecManager(kernel_dirs=[str(directory / "kernels")])
    return KernelManager(kernel_name="jump-filter-validation", kernel_spec_manager=manager)


def _callback_checks(export_directory):
    # TEST LOGIC: Exercise the callbacks after the original notebook has executed; paths are literals, never shell text.
    return '''# TEST LOGIC: Confirm native interactive chart objects and the initial applied snapshot.
import json
from pathlib import Path
import plotly.graph_objects as go
from jump_filter import METHODS
assert panel.result is not None, panel.status.value
assert panel.applied_config.method == "consensus"
assert panel.applied_scope == "selected"
assert panel.result.index.equals(data.loc[data["CUSIP"].eq(panel.cusip.value)].index)
assert len(panel.result) < len(data)
assert len(panel._figure_widgets) == 2
assert all(isinstance(figure, go.FigureWidget) and len(figure.data) > 0 for figure in panel._figure_widgets)
assert all("application/vnd.jupyter.widget-view+json" in figure._repr_mimebundle_() for figure in panel._figure_widgets)

# TEST LOGIC: Linked exact inputs and sliders synchronize in both directions.
panel.params["threshold"].value = 5.1
assert panel._param_sliders["threshold"].value == 5.1
panel._param_sliders["threshold"].value = 5.2
assert panel.params["threshold"].value == 5.2

# TEST LOGIC: Selecting a method updates pending help without mutating the applied review.
original_result = panel.result
panel.method.value = "rolling_iqr"
assert panel.result is original_result
assert panel.applied_config.method == "consensus"
assert panel._param_sliders["threshold"].disabled
assert not panel._param_sliders["iqr_multiplier"].disabled
panel.apply_button.click()
assert panel.result is not original_result, panel.status.value
assert panel.applied_config.method == "rolling_iqr"
applied_result = panel.result

# TEST LOGIC: First visits review one bond and revisits reuse its cached applied dataframe.
old_figure_ids = [figure.model_id for figure in panel._figure_widgets]
initial_bond = panel.cusip.value
panel.cusip.value = panel.cusip.options[1][1]
assert panel.result is not applied_result
assert panel.result["CUSIP"].eq(panel.cusip.value).all()
assert [figure.model_id for figure in panel._figure_widgets] != old_figure_ids
assert str(panel.cusip.value) in panel._figure_widgets[0].layout.title.text
panel.cusip.value = initial_bond
assert panel.result is applied_result

# TEST LOGIC: Invalid settings do not replace the successful applied result or its export configuration.
panel.params["min_neighbors"].value = 60
panel.apply_button.click()
assert panel.result is applied_result
assert panel.applied_config.min_neighbors == 6
assert "Could not apply settings" in panel.status.value
panel.params["min_neighbors"].value = 6

# TEST LOGIC: Method comparison uses the applied configuration and displays a native widget figure.
panel.compare_button.click()
assert panel.comparison is not None, panel.status.value
assert list(panel.comparison["method"]) == list(METHODS)
assert panel.comparison["total"].nunique() == 1
assert isinstance(panel._comparison_widget, go.FigureWidget)
assert panel.tabs.selected_index == 2

# TEST LOGIC: Export saves the applied review even when current method settings are pending.
panel.method.value = "local_piecewise"
panel.export_path.value = EXPORT_DIRECTORY
panel.export_button.click()
exported = sorted(Path(EXPORT_DIRECTORY).glob("review_*"))
assert len(exported) == 1, panel.status.value
settings = json.loads((exported[0] / "settings.json").read_text())
assert settings["config"]["method"] == "rolling_iqr"
assert settings["selected_cusip"] == str(panel.cusip.value)
assert settings["review_scope"] == "selected"
assert settings["rows"] == len(panel.result) < settings["source_rows"]
assert (exported[0] / "annotated_trades.csv").is_file()
assert (exported[0] / "method_comparison.csv").is_file()
assert (exported[0] / "selected_bond.html").is_file()

# TEST LOGIC: The explicit batch option evaluates all rows, with complete source order retained.
panel.scope.value = "all"
panel.apply_button.click()
assert panel.applied_scope == "all", panel.status.value
assert panel.result.index.equals(data.index)
batch_result = panel.result
panel.cusip.value = panel.cusip.options[1][1]
assert panel.result is batch_result
print("Notebook dashboard callback validation passed")
'''.replace("EXPORT_DIRECTORY", repr(str(export_directory)))


def test_shipped_notebook_executes_in_real_kernel_with_native_widget_outputs(tmp_path):
    # TEST LOGIC: A minimal engine installation skips this optional integration check cleanly.
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("ipywidgets")
    pytest.importorskip("anywidget")
    pytest.importorskip("plotly")

    # TEST LOGIC: Execute the exact shipped notebook with a real kernel and appended review interactions.
    notebook = nbformat.read(PROJECT / "jump_filter_dashboard.ipynb", as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(_callback_checks(tmp_path / "exports")))
    client = nbclient.NotebookClient(notebook, km=_kernel_manager(tmp_path), timeout=180,
                                   resources={"metadata": {"path": str(PROJECT)}}, store_widget_state=True)
    executed = client.execute()
    outputs = [output for cell in executed.cells if cell.cell_type == "code" for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert any("Notebook dashboard callback validation passed" in output.get("text", "") for output in outputs)
    assert any(WIDGET_VIEW in output.get("data", {}) for output in outputs)

    # TEST LOGIC: Native figures, formulas, and tables survive the actual Output-widget message protocol.
    states = executed.metadata.widgets["application/vnd.jupyter.widget-state+json"]["state"]
    assert any(entry["model_name"] == "AnyModel" and "_esm" in entry["state"] for entry in states.values())
    nested_outputs = [output for entry in states.values() for output in entry["state"].get("outputs", [])]
    assert sum(WIDGET_VIEW in output.get("data", {}) for output in nested_outputs) >= 3
    assert any("text/latex" in output.get("data", {}) for output in nested_outputs)
    html_outputs = [output["data"]["text/html"] for output in nested_outputs if "text/html" in output.get("data", {})]
    assert any("local_piecewise" in value for value in html_outputs)
    assert any("fitting coverage" in value for value in html_outputs)
    assert not any("Plotly.newPlot" in value for value in html_outputs)

    # TEST LOGIC: Execution remains in memory; the source notebook ships without persisted results or private records.
    original = nbformat.read(PROJECT / "jump_filter_dashboard.ipynb", as_version=4)
    assert all(cell.get("outputs", []) == [] for cell in original.cells)
    assert all(cell.get("execution_count") is None for cell in original.cells)
