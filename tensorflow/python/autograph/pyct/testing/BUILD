load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "codegen",
    srcs = [
        "codegen.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct",
        "//third_party/py/numpy",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "basic_definitions",
    srcs = ["basic_definitions.py"],
    visibility = ["//visibility:public"],
)

py_strict_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "//third_party/py/numpy",
    ],
)
