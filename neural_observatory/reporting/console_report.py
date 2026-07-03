"""
Neural Observatory — Console Reporter
"""
from __future__ import annotations

import datetime
import os
import re
import sys
from typing import Optional

from .base import BaseReporter, Report
from ..analysis.base import AnalysisResult, AnalysisStatus
from ..core.configuration import ObservatoryConfig


# Disable colors if we're not in a TTY or if NO_COLOR is set
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:   return _c("32", t)
def yellow(t: str) -> str:  return _c("33", t)
def red(t: str) -> str:     return _c("31", t)
def cyan(t: str) -> str:    return _c("36", t)
def bold(t: str) -> str:    return _c("1",  t)
def dim(t: str) -> str:     return _c("2",  t)


def _visible_len(s: str) -> int:
    """Length of string after stripping ANSI escape codes."""
    return len(_ANSI_RE.sub("", s))


def _pad_center(s: str, width: int) -> str:
    """Center string within width, accounting for invisible ANSI chars."""
    visible = _visible_len(s)
    total_pad = max(width - visible, 0)
    left = total_pad // 2
    right = total_pad - left
    return " " * left + s + " " * right


STATUS_ICON = {
    AnalysisStatus.HEALTHY:  green("●"),
    AnalysisStatus.INFO:     cyan("ℹ"),
    AnalysisStatus.WARNING:  yellow("⚠"),
    AnalysisStatus.CRITICAL: red("✖"),
}

STATUS_COLOR = {
    AnalysisStatus.HEALTHY:  green,
    AnalysisStatus.INFO:     cyan,
    AnalysisStatus.WARNING:  yellow,
    AnalysisStatus.CRITICAL: red,
}


class ConsoleReporter(BaseReporter):
    def report(
        self,
        report: Report,
        config: Optional[ObservatoryConfig] = None,
    ) -> None:
        cfg = config or ObservatoryConfig()

        results = report.filtered(
            min_severity=cfg.report_min_severity,
            max_results=cfg.report_max_layers,
            include_healthy=cfg.report_include_healthy,
        )

        self._print_header(report)

        if not results:
            print(green("  All layers healthy — no issues detected.\n"))
            return

        by_analyzer = {}
        for r in results:
            by_analyzer.setdefault(r.analyzer_name, []).append(r)

        for analyzer_name, analyzer_results in sorted(by_analyzer.items()):
            self._print_section(analyzer_name, analyzer_results, cfg)

        self._print_summary(report)

    def _print_header(self, report: Report) -> None:
        ts = datetime.datetime.fromtimestamp(report.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        overall = report.overall_status()
        color = STATUS_COLOR.get(overall, lambda x: x)

        n_warn = len(report.warnings())
        n_crit = len(report.critical())

        width = 60
        border = "═" * width
        title = "NEURAL OBSERVATORY REPORT"
        sub   = f"Step {report.step}  ·  Epoch {report.epoch}  ·  {ts}"
        stat  = f"{n_warn} warning(s)  ·  {n_crit} critical"

        print()
        print(f"╔{border}╗")
        print(f"║{title:^{width}}║")
        print(f"║{sub:^{width}}║")
        print(f"║{_pad_center(color(stat), width)}║")
        print(f"╚{border}╝")
        print()

    def _print_section(
        self,
        analyzer_name: str,
        results: list,
        cfg: ObservatoryConfig,
    ) -> None:
        print(bold(analyzer_name.upper().replace("_", " ")))
        print("─" * 60)

        for r in results:
            self._print_result_row(r, cfg)

        print()

    def _print_result_row(
        self,
        r: AnalysisResult,
        cfg: ObservatoryConfig,
    ) -> None:
        icon   = STATUS_ICON[r.status]
        color  = STATUS_COLOR[r.status]
        status = color(r.status.value.upper())

        metric_str = self._format_key_metrics(r)

        name_col = r.layer_name[:30].ljust(30)
        metrics_col = metric_str[:20].ljust(20)

        line = f"  {name_col}  {metrics_col}  {icon} {status}"
        print(line)

        for w in r.warnings:
            print(f"      {yellow('▶')} {w}")

        if cfg.verbose:
            for i in r.info:
                print(f"      {dim(i)}")

    def _format_key_metrics(self, r: AnalysisResult) -> str:
        m = r.metrics
        parts = []

        if "dead_ratio" in m:
            parts.append(f"dead={m['dead_ratio']:.0%}")
        if "mean_norm" in m:
            parts.append(f"norm={m['mean_norm']:.4f}")
        if "sparsity" in m:
            parts.append(f"sparse={m['sparsity']:.0%}")
        if "drift_from_init" in m:
            parts.append(f"drift={m['drift_from_init']:.1%}")
        if "total_anomalies" in m and m["total_anomalies"] > 0:
            parts.append(f"anomalies={int(m['total_anomalies'])}")

        return ", ".join(parts[:2])

    def _print_summary(self, report: Report) -> None:
        s = report.summary_dict()
        overall = report.overall_status()
        color = STATUS_COLOR.get(overall, lambda x: x)

        print("─" * 60)
        print(
            f"  Overall: {color(bold(overall.value.upper()))}"
            f"  |  severity={s['max_severity']:.2f}"
            f"  |  {s['status_counts']['critical']} critical"
            f", {s['status_counts']['warning']} warnings"
            f", {s['status_counts']['healthy']} healthy"
        )
        print()