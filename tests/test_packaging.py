"""Tests for packaging correctness and configuration validation."""

import importlib
import shutil
from pathlib import Path

import pytest


def test_discovery_excludes_workspace_artifacts(tmp_path):
    from setuptools.config.pyprojecttoml import read_configuration

    root = Path(__file__).parent.parent
    shutil.copy2(root / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "README.md").write_text("Packaging fixture", encoding="utf-8")
    for relative in (
        "neural_observatory",
        "neural_observatory/hooks",
        "venv/unrelated",
        "build/unrelated",
        "outputs/unrelated",
    ):
        package = tmp_path / relative
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    config = read_configuration(str(tmp_path / "pyproject.toml"))
    assert set(config["tool"]["setuptools"]["packages"]) == {
        "neural_observatory",
        "neural_observatory.hooks",
    }


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
    discovered = set(
        find_packages(
            where=str(root), exclude=["tests*", "examples*", "docs*"]
        )
    )
    assert set(SUBPACKAGES) <= discovered, (
        f"find_packages() did not discover all subpackages: "
        f"missing {set(SUBPACKAGES) - discovered}"
    )


def test_pyproject_does_not_use_a_bare_explicit_package_list():
    """Guard against the incomplete explicit-package configuration."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    raw_text = pyproject_path.read_text()

    # Comments can legitimately mention the old configuration.
    text = "\n".join(
        line
        for line in raw_text.splitlines()
        if not line.strip().startswith("#")
    )
    normalized = text.replace(" ", "")

    assert (
        'packages=["neural_observatory"]' not in normalized
    ), "pyproject.toml appears to use a bare explicit single-package list."
    assert (
        "[tool.setuptools.packages.find]" in text
    ), "Expected [tool.setuptools.packages.find] to be present."
