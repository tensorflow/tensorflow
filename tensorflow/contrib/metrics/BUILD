# Description:
#   Contains ops for evaluation metrics and summary statistics.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku,ptucker): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_set_ops.so",
    srcs = [
        "ops/set_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/metrics/kernels:set_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["set_ops"],
)

tf_gen_op_wrapper_py(
    name = "set_ops",
    hidden = [
        "DenseToDenseSetOperation",
        "DenseToSparseSetOperation",
        "SparseToSparseSetOperation",
        "SetSize",
    ],
    deps = ["//tensorflow/contrib/metrics:set_ops_op_lib"],
)

tf_kernel_library(
    name = "set_ops_kernels",
    deps = [
        "//tensorflow/contrib/metrics/kernels:set_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

py_library(
    name = "metrics_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]) + glob(["python/metrics/*.py"]),
    data = [":python/ops/_set_ops.so"],
    srcs_version = "PY2AND3",
    deps = [":set_ops"],
)

py_test(
    name = "set_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/set_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "classification_test",
    srcs = ["python/metrics/classification_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "confusion_matrix_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/confusion_matrix_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "histogram_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/histogram_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "metric_ops_util_test",
    srcs = ["python/ops/metric_ops_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "metric_ops_test",
    srcs = ["python/ops/metric_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
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
