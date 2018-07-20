# Description:
#   Contains library to create sequence-to-sequence models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "gpu_py_tests")

py_library(
    name = "seq2seq_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ] + glob(
        ["python/ops/*.py"],
        exclude = ["python/ops/**/*_test.py"],
    ),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

gpu_py_tests(
    name = "seq2seq_test",
    size = "medium",
    srcs = ["python/kernel_tests/seq2seq_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:rnn",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "noasan",  # times out b/63678675
        "optonly",  # times out (flaky)
    ],
)
