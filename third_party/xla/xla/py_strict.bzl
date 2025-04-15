"""Default (OSS) build versions of Python strict rules."""

load("@local_xla//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)
