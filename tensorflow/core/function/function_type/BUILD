load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    licenses = ["notice"],
)

pytype_strict_library(
    name = "function_type",
    srcs = [
        "function_type.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/types",
    ],
)

py_strict_test(
    name = "function_type_test",
    srcs = ["function_type_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
        "nopip",
    ],
    deps = [
        ":function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)
