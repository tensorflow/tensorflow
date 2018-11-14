package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "range_benchmark",
    srcs = ["range_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)
