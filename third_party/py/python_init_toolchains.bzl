"""
Legacy-compatible wrapper for python toolchain registration.
See the WORKSPACE file for instructions on using the updated
python toolchain registration.
"""

load(
    "@rules_ml_toolchain//py:python_register_toolchain.bzl",
    "python_register_toolchain",
)

def python_init_toolchains(name = "python", python_version = None, **kwargs):
    python_register_toolchain(
        name = name,
        python_version = python_version,
        **kwargs
    )
