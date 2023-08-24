licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
load("@python//:defs.bzl", "interpreter")

py_runtime(
    name = "py2_runtime",
    interpreter_path = interpreter,
    python_version = "PY2",
)

py_runtime(
    name = "py3_runtime",
    interpreter_path = interpreter,
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = ":py2_runtime",
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
    target_compatible_with = [%{PLATFORM_CONSTRAINT}],
    exec_compatible_with = [%{PLATFORM_CONSTRAINT}],
)

alias(name = "python_headers",
      actual = "@python//:python_headers")

# This alias is exists for the use of targets in the @llvm-project dependency,
# which expect a python_headers target called @python_runtime//:headers. We use
# a repo_mapping to alias python_runtime to this package, and an alias to create
# the correct target.
alias(
    name = "headers",
    actual = ":python_headers",
)


config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)