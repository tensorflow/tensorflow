# Description:
#   A Cudnn RNN wrapper.
#   APIs are meant to change over time.
package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

tf_custom_op_library(
    name = "python/ops/_cudnn_rnn_ops.so",
    srcs = [
        "kernels/cudnn_rnn_ops.cc",
        "ops/cudnn_rnn_ops.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["cudnn_rnn_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "cudnn_rnn_ops",
    deps = [":cudnn_rnn_ops_op_lib"],
)

py_library(
    name = "cudnn_rnn_py",
    srcs = [
        "__init__.py",
        "python/ops/cudnn_rnn_ops.py",
    ],
    data = [
        ":python/ops/_cudnn_rnn_ops.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cudnn_rnn_ops",
    ],
)

cuda_py_test(
    name = "cudnn_rnn_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/cudnn_rnn_ops_test.py"],
    additional_deps = [
        ":cudnn_rnn_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
        "requires_cudnn5",
    ],
)

cuda_py_test(
    name = "cudnn_rnn_ops_benchmark",
    size = "large",
    srcs = ["python/kernel_tests/cudnn_rnn_ops_benchmark.py"],
    additional_deps = [
        ":cudnn_rnn_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
        "requires_cudnn5",
    ],
)

cc_test(
    name = "cudnn_rnn_ops_test_cc",
    size = "small",
    srcs = [
        "ops/cudnn_rnn_ops_test.cc",
    ],
    deps = [
        ":cudnn_rnn_ops_op_lib",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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
