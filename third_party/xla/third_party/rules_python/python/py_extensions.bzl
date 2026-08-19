"""Wrapper for py_extensions in OSS using native.cc_binary."""

load(":py_extension.bzl", _py_extension = "py_extension")

py_extension = _py_extension
py_extensions = _py_extension
