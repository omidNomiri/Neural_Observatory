"""
Neural Observatory — JSON Reporter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Optional, Union

from .base import BaseReporter, Report
from ..core.configuration import ObservatoryConfig


class JSONReporter(BaseReporter):
    def __init__(
        self,
        output: Union[str, Path, IO, None] = None,
        indent: Optional[int] = 2,
        append: bool = False,
    ) -> None:
        self._output = output
        self._indent = indent
        self._append = append

    def report(
        self,
        report: Report,
        config: Optional[ObservatoryConfig] = None,
    ) -> None:
        cfg = config or ObservatoryConfig()
        payload = self._build_payload(report, cfg)
        
        # If appending, we don't care about pretty indentation, we just want valid JSONL
        if self._append:
            json_str = json.dumps(payload, default=str)
        else:
            json_str = json.dumps(payload, indent=self._indent, default=str)

        if self._output is None:
            print(json_str)
            return

        if isinstance(self._output, (str, Path)):
            mode = "a" if self._append else "w"
            with open(self._output, mode, encoding="utf-8") as fh:
                fh.write(json_str + "\n")
        else:
            self._output.write(json_str + "\n")

    @staticmethod
    def _build_payload(report: Report, cfg: ObservatoryConfig) -> dict:
        results = report.filtered(
            min_severity=cfg.report_min_severity,
            max_results=cfg.report_max_layers,
            include_healthy=cfg.report_include_healthy,
        )

        return {
            "meta": {
                "timestamp": report.timestamp,
                "step": report.step,
                "epoch": report.epoch,
            },
            "summary": report.summary_dict(),
            "results": [
                {
                    "analyzer": r.analyzer_name,
                    "layer": r.layer_name,
                    "status": r.status.value,
                    "severity": r.severity,
                    "metrics": r.metrics,
                    "warnings": r.warnings,
                    "info": r.info,
                    "step": r.step,
                }
                for r in results
            ],
        }