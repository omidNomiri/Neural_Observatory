import os
import sys

# Add the project root to sys.path so Sphinx can find the module
sys.path.insert(0, os.path.abspath('..'))

project = 'Neural Observatory'
author = 'Omid Nomiri'
release = '0.5.0'

# -- General options ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Pulls docstrings from Python code
    'sphinx.ext.napoleon',  # Supports Google and NumPy style docstrings
    'myst_parser',         # Allows writing in Markdown
    'sphinx.ext.viewcode',  # Link to the source code
]

# -- Options for HTML output -------------------------------------------
html_theme = 'furo'

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

# -- myst_parser options -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Suppress warnings about duplicate object descriptions
suppress_warnings = ['duplicate_object']
