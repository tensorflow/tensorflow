"""Wrapper for py_binary rule."""

load("@rules_python//python:defs.bzl", _py_binary = "py_binary")
load("//third_party/rules_python/python:common.bzl", "filter_kwargs")

def py_binary(**kwargs):
    """Wrapper for py_binary that strictly filters supported attributes."""
    _py_binary(**filter_kwargs(kwargs))
