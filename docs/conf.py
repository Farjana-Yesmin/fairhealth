import os, sys
sys.path.insert(0, os.path.abspath(".."))

project   = "FairHealth"
author    = "Farjana Yesmin"
release   = "0.1.0"
copyright = "2026, Farjana Yesmin"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"
napoleon_google_docstring = True
