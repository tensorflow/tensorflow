"""Exported Python rules wrappers."""

load(":py_binary.bzl", _py_binary = "py_binary")
load(":py_library.bzl", _py_library = "py_library")
load(":py_test.bzl", _py_test = "py_test")

py_library = _py_library
py_binary = _py_binary
py_test = _py_test
