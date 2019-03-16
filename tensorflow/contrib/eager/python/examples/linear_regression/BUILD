licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

py_binary(
    name = "linear_regression",
    srcs = ["linear_regression.py"],
    srcs_version = "PY2AND3",
    deps = [":linear_regression_lib"],
)

py_library(
    name = "linear_regression_lib",
    srcs = ["linear_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
    ],
)

cuda_py_test(
    name = "linear_regression_test",
    size = "small",
    srcs = ["linear_regression_test.py"],
    additional_deps = [
        ":linear_regression_lib",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
        "oss_serial",
    ],
)

cuda_py_test(
    name = "linear_regression_graph_test",
    size = "small",
    srcs = ["linear_regression_graph_test.py"],
    additional_deps = [
        ":linear_regression_lib",
        "//tensorflow:tensorflow_py",
    ],
)
