package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

py_library(
    name = "avro_utils",
    srcs = [
        "avro_serialization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)