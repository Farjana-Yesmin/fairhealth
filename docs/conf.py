import os
import sys

# Point to the package root
sys.path.insert(0, os.path.abspath('..'))

project   = 'FairHealth'
author    = 'Farjana Yesmin'
release   = '0.1.0'
copyright = '2026, Farjana Yesmin'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
napoleon_google_docstring = True
autodoc_mock_imports = [
    'torch', 'tenseal', 'shap', 'lime',
    'skfuzzy', 'xgboost', 'wfdb', 'ucimlrepo',
    'sklearn', 'scipy', 'matplotlib', 'pandas', 'numpy',
]
