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
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

tf_custom_op_library(
    name = "python/ops/_image_ops.so",
    srcs = [
        "kernels/image_ops.cc",
        "kernels/image_ops.h",
        "ops/image_ops.cc",
    ],
    gpu_srcs = [
        "kernels/image_ops_gpu.cu.cc",
        "kernels/image_ops.h",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["image_ops"],
)

tf_gen_op_wrapper_py(
    name = "image_ops",
    deps = [":image_ops_op_lib"],
)

py_library(
    name = "image_py",
    srcs = [
        "__init__.py",
        "python/ops/image_ops.py",
    ],
    data = [":python/ops/_image_ops.so"],
    srcs_version = "PY2AND3",
    deps = [
        ":image_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib:contrib_py",
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
