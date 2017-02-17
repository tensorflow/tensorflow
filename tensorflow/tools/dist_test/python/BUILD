# Python tools for running distributed benchmarks.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "benchmark_util_lib",
    srcs = ["benchmark_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "benchmark_util_test",
    srcs = ["benchmark_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_util_lib",
        "//tensorflow/python:client_testlib",
    ],
)
