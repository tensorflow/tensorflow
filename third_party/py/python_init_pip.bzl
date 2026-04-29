"""
Legacy-compatible wrapper for Hermetic Python initialization.
See the WORKSPACE file for instructions on using the updated
Hermetic Python initialization.
"""

load(
    "@rules_ml_toolchain//py:python_init_pip.bzl",
    _python_init_pip = "python_init_pip",
)

def python_init_pip():
    _python_init_pip()
