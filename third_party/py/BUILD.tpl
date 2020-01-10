licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "py2_runtime",
    interpreter_path = "%{PYTHON_BIN_PATH}",
    python_version = "PY2",
)

py_runtime(
    name = "py3_runtime",
    interpreter_path = "%{PYTHON_BIN_PATH}",
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
)

# To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
# See https://docs.python.org/3/extending/windows.html
cc_import(
    name = "python_lib",
    interface_library = select({
        ":windows": ":python_import_lib",
        # A placeholder for Unix platforms which makes --no_build happy.
        "//conditions:default": "not-existing.lib",
    }),
    system_provided = 1,
)

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    deps = select({
        ":windows": [":python_lib"],
        "//conditions:default": [],
    }),
    includes = ["python_include"],
)

cc_library(
    name = "numpy_headers",
    hdrs = [":numpy_include"],
    includes = ["numpy_include"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

%{PYTHON_INCLUDE_GENRULE}
%{NUMPY_INCLUDE_GENRULE}
%{PYTHON_IMPORT_LIB_GENRULE}
