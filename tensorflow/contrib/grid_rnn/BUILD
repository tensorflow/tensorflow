# Description:
#   Contains classes to construct GridRNN cells
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "gpu_py_tests")

py_library(
    name = "grid_rnn_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variable_scope",
    ],
)

gpu_py_tests(
    name = "grid_rnn_test",
    srcs = ["python/kernel_tests/grid_rnn_test.py"],
    additional_deps = [
        ":grid_rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)
