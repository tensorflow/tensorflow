# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "seq2seq_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
)

cuda_py_test(
    name = "decoder_fn_test",
    size = "medium",
    srcs = ["python/kernel_tests/decoder_fn_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "seq2seq_test",
    size = "medium",
    srcs = ["python/kernel_tests/seq2seq_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
