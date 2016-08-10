# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_sdca_ops.so",
    srcs = [
        "ops/sdca_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/linear_optimizer/kernels:sdca_ops",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["sdca_ops"],
)

tf_gen_op_wrapper_py(
    name = "sdca_ops",
    deps = ["//tensorflow/contrib/linear_optimizer:sdca_ops_op_lib"],
)

tf_kernel_library(
    name = "sdca_op_kernels",
    deps = [
        "//tensorflow/contrib/linear_optimizer/kernels:sdca_ops",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

py_library(
    name = "sdca_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/sdca_ops.py",
        "python/sdca_optimizer.py",
    ],
    data = [":python/ops/_sdca_ops.so"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops",
        "//tensorflow/contrib/lookup:lookup_py",
    ],
)

py_test(
    name = "sdca_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/sdca_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops_py",
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
)
