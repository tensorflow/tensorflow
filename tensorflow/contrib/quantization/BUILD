# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",  # @unused
    "tf_gen_op_wrapper_py",  # @unused
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "quantization_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow/python:array_ops_gen",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "python/array_ops.py",
        "python/math_ops.py",
        "python/nn_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:common_shapes",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops_gen",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)
