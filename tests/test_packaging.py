"""Tests for packaging correctness and configuration validation."""
import importlib
import os
from pathlib import Path

import pytest


SUBPACKAGES = [
    "neural_observatory",
    "neural_observatory.core",
    "neural_observatory.hooks",
    "neural_observatory.collectors",
    "neural_observatory.storage",
    "neural_observatory.analysis",
    "neural_observatory.reporting",
]


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_is_importable(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_find_packages_discovers_every_subpackage():
    """Guards against regressions where subpackages are silently dropped."""
    from setuptools import find_packages
    root = Path(__file__).parent.parent
    discovered = set(find_packages(where=str(root), exclude=["tests*", "examples*", "docs*"]))
    assert set(SUBPACKAGES) <= discovered, (
        f"find_packages() did not discover all subpackages: "
        f"missing {set(SUBPACKAGES) - discovered}"
    )


def test_pyproject_does_not_use_a_bare_explicit_package_list():
    """Statically guard against re-introducing the bare `packages = [...]` bug."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    raw_text = pyproject_path.read_text()
    
    # Strip comments so documenting the old bug in a comment doesn't break this test
    text = "\n".join(
        line for line in raw_text.splitlines() if not line.strip().startswith("#")
    )
    normalized = text.replace(" ", "")

    assert 'packages=["neural_observatory"]' not in normalized, (
        "pyproject.toml appears to use a bare explicit single-package list."
    )
    assert "[tool.setuptools.packages.find]" in text, (
        "Expected [tool.setuptools.packages.find] to be present."
    )