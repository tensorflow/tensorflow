"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
load("@rules_python//python:versions.bzl", "MINOR_MAPPING")

def python_init_toolchains():
    if HERMETIC_PYTHON_VERSION in MINOR_MAPPING:
        python_register_toolchains(
            name = "python",
            ignore_root_user_error = True,
            python_version = HERMETIC_PYTHON_VERSION,
        )
