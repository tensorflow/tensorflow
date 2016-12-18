# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_test",
    "tf_py_test",
    "tf_gen_op_libs",
    "tf_kernel_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

py_library(
    name = "sparsemax_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [
        ":python/ops/_sparsemax.so",
        ":python/ops/_sparsemax_loss.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_tests(
    name = "sparsemax_test",
    size = "small",
    srcs = ["python/kernel_tests/sparsemax_test.py"],
    additional_deps = [
        ":sparsemax_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "sparsemax_loss_test",
    size = "medium",
    srcs = ["python/kernel_tests/sparsemax_loss_test.py"],
    additional_deps = [
        ":sparsemax_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_custom_op_library(
    name = "python/ops/_sparsemax.so",
    srcs = [
        "kernels/sparsemax_functor.cc",
        "kernels/sparsemax_functor.h",
        "kernels/sparsemax.cc",
        "ops/sparsemax.cc",
    ],
    gpu_srcs = [
        "kernels/sparsemax_functor.cu.cc",
        "kernels/sparsemax_functor.h",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

tf_custom_op_library(
    name = "python/ops/_sparsemax_loss.so",
    srcs = [
        "kernels/sparsemax_loss.cc",
        "kernels/sparsemax_loss.h",
        "ops/sparsemax_loss.cc",
    ],
    gpu_srcs = [
        "kernels/sparsemax_loss.cu.cc",
        "kernels/sparsemax_loss.h",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
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

tf_gen_op_libs(
    op_lib_names = [
        "sparsemax",
        "sparsemax_loss",
    ],
)

tf_kernel_library(
    name = "sparsemax_kernels",
    srcs = [
        "kernels/sparsemax_functor.cc",
        "kernels/sparsemax_functor.h",
        "kernels/sparsemax.cc",
    ],
    gpu_srcs = [
        "kernels/sparsemax_functor.cu.cc",
        "kernels/sparsemax_functor.h",
    ],
    prefix = "kernels/sparsemax_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)


tf_kernel_library(
    name = "sparsemax_loss_kernels",
    srcs = [
        "kernels/sparsemax_loss.cc",
        "kernels/sparsemax_loss.h",
    ],
    gpu_srcs = [
        "kernels/sparsemax_loss.cu.cc",
        "kernels/sparsemax_loss.h",
    ],
    prefix = "kernels/sparsemax_loss_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)
