# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

tf_custom_op_library(
    name = "python/ops/_lstm_ops.so",
    srcs = [
        "ops/lstm_ops.cc",
        "kernels/lstm_ops.cc",
        "kernels/lstm_ops.h",
    ],
    gpu_srcs = [
        "kernels/lstm_ops_gpu.cu.cc",
        "kernels/lstm_ops.h",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["lstm_ops"],
)

tf_gen_op_wrapper_py(
    name = "lstm_ops",
    deps = [":lstm_ops_op_lib"],
    hidden = ["LSTMCellBlockGrad"],
    require_shape_functions = True,
)

py_library(
    name = "rnn_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    data = [":python/ops/_lstm_ops.so"],
    deps = [":lstm_ops",],
)

cuda_py_tests(
    name = "rnn_tests",
    size = "small",
    srcs = glob(["python/kernel_tests/*.py"]),
    additional_deps = [
        ":rnn_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:kernel_tests/rnn_test_lib",
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
