package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

load("//tensorflow:tensorflow.bzl", "py_test")

licenses(["notice"])  # Apache 2.0

py_library(
    name = "distributions",
    srcs = glob(["*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:util",
    ],
)

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
