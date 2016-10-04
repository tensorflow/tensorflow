# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "seq2seq_py",
    srcs = [
        "__init__.py",
        "python/seq2seq/__init__.py",
        "python/seq2seq/loss_ops.py",
        "python/seq2seq/layers_ops.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "loss_ops_test",
    srcs = glob(["python/kernel_test/loss_ops_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":seq2seq_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "layers_ops_test",
    srcs = glob(["python/kernel_test/layers_ops_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":seq2seq_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)
