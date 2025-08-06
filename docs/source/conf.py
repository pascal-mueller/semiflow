import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "Semiflow"
copyright = "2025, Pascal D. Müller"
author = "Pascal D. Müller"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_numpy_docstring = True
