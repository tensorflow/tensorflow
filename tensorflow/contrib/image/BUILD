# Description:
#   Contains ops for image manipulation.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_image_ops.so",
    srcs = [
        "kernels/bipartite_match_op.cc",
        "kernels/image_ops.cc",
        "kernels/image_ops.h",
        "ops/image_ops.cc",
    ],
    gpu_srcs = [
        "kernels/image_ops_gpu.cu.cc",
        "kernels/image_ops.h",
    ],
)

tf_kernel_library(
    name = "image_ops_kernels",
    srcs = [
        "kernels/bipartite_match_op.cc",
        "kernels/image_ops.cc",
        "kernels/image_ops.h",
    ],
    gpu_srcs = [
        "kernels/image_ops_gpu.cu.cc",
        "kernels/image_ops.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["image_ops"],
)

tf_gen_op_wrapper_py(
    name = "image_ops",
    deps = [":image_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "image_py",
    srcs = [
        "__init__.py",
        "python/ops/image_ops.py",
    ],
    dso = [":python/ops/_image_ops.so"],
    kernels = [
        ":image_ops_kernels",
        ":image_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":image_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/image_ops_test.py"],
    additional_deps = [
        ":image_py",
        ":single_image_random_dot_stereograms_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_custom_op_library(
    name = "python/ops/_single_image_random_dot_stereograms.so",
    srcs = [
        "kernels/single_image_random_dot_stereograms_ops.cc",
        "ops/single_image_random_dot_stereograms_ops.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["single_image_random_dot_stereograms_ops"],
)

tf_gen_op_wrapper_py(
    name = "single_image_random_dot_stereograms_ops",
    deps = [":single_image_random_dot_stereograms_ops_op_lib"],
)

py_library(
    name = "single_image_random_dot_stereograms_py",
    srcs = glob(["python/ops/single*.py"]) + ["__init__.py"],
    data = [":python/ops/_single_image_random_dot_stereograms.so"],
    srcs_version = "PY2AND3",
    deps = [
        ":image_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
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
