"""Wrapper for py_library rule."""

load("@rules_python//python:defs.bzl", _py_library = "py_library")
load("//third_party/rules_python/python:common.bzl", "filter_kwargs")

def py_library(**kwargs):
    """Wrapper for py_library that strictly filters supported attributes."""
    _py_library(**filter_kwargs(kwargs))
