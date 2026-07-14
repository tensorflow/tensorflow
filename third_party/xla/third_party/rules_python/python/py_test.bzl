"""Wrapper for py_test rule."""

load("@rules_python//python:defs.bzl", _py_test = "py_test")
load("//third_party/rules_python/python:common.bzl", "filter_kwargs")

def py_test(**kwargs):
    """Wrapper for py_test that strictly filters supported attributes."""
    _py_test(**filter_kwargs(kwargs))
