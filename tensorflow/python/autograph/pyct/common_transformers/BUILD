load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:gast_util",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":common_transformers",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:loader",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)
