licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "lang",
    srcs = [
        "directives.py",
        "special_functions.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/operators",
    ],
)

py_test(
    name = "special_functions_test",
    srcs = ["special_functions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lang",
        "//tensorflow/python:client_testlib",
    ],
)
