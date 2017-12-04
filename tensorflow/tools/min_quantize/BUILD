# Description:
#   Utilities for quantizing TensorFlow graphs without additional Ops

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "quantize",
    srcs = ["quantize.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/min_quantize:obfuscate_and_quantize_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "obfuscate",
    srcs = ["obfuscate.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/min_quantize:obfuscate_and_quantize_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
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
