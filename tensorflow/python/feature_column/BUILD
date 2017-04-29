package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

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
    name = "feature_column",
    srcs = ["feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "feature_column_test",
    srcs = ["feature_column_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
    ],
)
