# Tools for running distributed benchmarks.

licenses(["notice"])  # Apache 2.0

exports_files(["k8s_tensorflow.py"])

py_library(
    name = "k8s_tensorflow_lib",
    srcs = ["k8s_tensorflow_lib.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "k8s_tensorflow_test",
    srcs = ["k8s_tensorflow_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":k8s_tensorflow_lib",
        "//tensorflow/python:client_testlib",
    ],
)
