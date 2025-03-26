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
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:templates",
        "//third_party/py/numpy",
        "@pypi_gast//:pkg",
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
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/platform:client_testlib",
    ],
)
