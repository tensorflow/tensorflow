# Description:
#   Contains classes to construct a CRF layer
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "crf_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//third_party/py/numpy",
    ],
)

cuda_py_tests(
    name = "crf_test",
    srcs = ["python/kernel_tests/crf_test.py"],
    additional_deps = [
        ":crf_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)
