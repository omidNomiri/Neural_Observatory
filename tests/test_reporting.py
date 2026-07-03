"""Tests for the reporting layer."""
import io
import contextlib
import json
import re

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.reporting import ConsoleReporter, JSONReporter, HTMLReporter
from conftest import run_training_steps


def _make_report(tiny_model):
    obs = Observatory(tiny_model, config=ObservatoryConfig(max_observations=50))
    obs.watch()
    run_training_steps(obs, tiny_model, n_steps=4)
    return obs.report()


def test_json_and_console_reporters_agree_on_filtered_results(tiny_model, capsys, tmp_path):
    report = _make_report(tiny_model)
    cfg = ObservatoryConfig()

    json_path = tmp_path / "report.json"
    JSONReporter(output=str(json_path)).report(report, cfg)
    payload = json.loads(json_path.read_text())

    expected = report.filtered(
        min_severity=cfg.report_min_severity,
        max_results=cfg.report_max_layers,
        include_healthy=cfg.report_include_healthy,
    )
    assert len(payload["results"]) == len(expected)
    assert payload["summary"]["total_results"] == len(report.results)


def test_console_reporter_alignment_is_ansi_aware(tiny_model, monkeypatch):
    """Regression test: header box borders must stay aligned with color codes."""
    import neural_observatory.reporting.console_report as console_mod
    monkeypatch.setattr(console_mod, "_USE_COLOR", True)

    report = _make_report(tiny_model)
    reporter = ConsoleReporter()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reporter.report(report, ObservatoryConfig())
    output = buf.getvalue()

    ansi_re = re.compile(r"\033\[[0-9;]*m")
    lines = [l for l in output.splitlines() if l.startswith("║")]
    visible_widths = {len(ansi_re.sub("", l)) for l in lines}

    assert len(visible_widths) == 1, (
        f"Header box lines have inconsistent visible widths: {visible_widths}"
    )


def test_html_reporter_escapes_layer_names_and_warnings(tiny_model, tmp_path):
    report = _make_report(tiny_model)
    out_path = tmp_path / "report.html"
    HTMLReporter(output_path=str(out_path)).report(report, ObservatoryConfig())
    html_text = out_path.read_text()

    assert "<!DOCTYPE html>" in html_text
    assert "<script>" not in html_text.lower()


def test_html_reporter_summary_card_has_no_crash_on_all_statuses(tiny_model, tmp_path):
    """Regression check for unused `bg`-variable cleanup in _summary_card."""
    report = _make_report(tiny_model)
    out_path = tmp_path / "report.html"
    HTMLReporter(output_path=str(out_path)).report(report, ObservatoryConfig())
    html_text = out_path.read_text()
    assert "Overall Status" in html_text