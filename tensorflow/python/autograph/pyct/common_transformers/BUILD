load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:gast_util",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    strict_deps = True,
    tags = ["no_oss"],
    deps = [
        ":common_transformers",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:loader",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)
