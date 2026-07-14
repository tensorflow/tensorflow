# Compatibility Wrappers for rules_python

This directory contains wrappers around the standard `rules_python` rules
(`py_library`, `py_binary`, `py_test`).

## Purpose

These wrappers allow existing Python rules (which may contain non-standard or
legacy attributes) to be built with standard open-source `rules_python` without
modification to the source `BUILD` files.

## Mechanism

Each wrapper (`py_library.bzl`, `py_binary.bzl`, `py_test.bzl`) imports the
corresponding upstream rule and wraps it. A shared utility in `common.bzl`
filters out known unsupported attributes before passing arguments to the
upstream rule.

## Attribute Support

### Stripped Attributes (Unsupported)

The following attributes are **automatically removed** to prevent build errors:

*   `strict_deps`
*   `lazy_imports`
*   `linking_mode`
*   `flaky_test_attempts` (for `py_test`)

## Usage

In your `BUILD` files, load these rules instead of the standard ones:

```starlark
load(
    "//third_party/rules_python/python:defs.bzl",
    "py_binary",
    "py_library",
    "py_test",
)
```