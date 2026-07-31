import os
import sys

# Add the project root to sys.path so Sphinx can find the module
sys.path.insert(0, os.path.abspath('..'))

project = 'Neural Observatory'
author = 'Omid Nomiri'
release = '0.3.0'

# -- General options ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Pulls docstrings from Python code
    'sphinx.ext.napoleon', # Supports Google and NumPy style docstrings
    'myst_parser',         # Allows writing in Markdown
]

# -- Options for HTML output -------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']

# -- myst_parser options -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}