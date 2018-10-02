package(
    default_visibility = ["//tensorflow:internal"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

licenses(["notice"])  # Apache 2.0

py_library(
    name = "distributions",
    srcs = glob(
        ["*.py"],
        exclude = ["util.py"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:tensor_util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
