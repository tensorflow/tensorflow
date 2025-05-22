"""Shims for loading the plain Python rules.

These are used to make internal/external code transformations managable. Once
Tensorflow is loading the Python rules directly from rules_python, these shims
can be removed.
"""

load("@rules_python//python:py_binary.bzl", _py_binary = "py_binary")
load("@rules_python//python:py_library.bzl", _py_library = "py_library")
load("@rules_python//python:py_test.bzl", _py_test = "py_test")

py_test = _py_test

py_binary = _py_binary

py_library = _py_library
