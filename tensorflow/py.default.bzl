"""Shims for loading the plain Python rules.

These are used to make internal/external code transformations managable. Once
Tensorflow is loading the Python rules directly from rules_python, these shims
can be removed.
"""

py_test = native.py_test

py_binary = native.py_binary

py_library = native.py_library
