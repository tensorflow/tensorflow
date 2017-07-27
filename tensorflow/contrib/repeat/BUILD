licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_repeat_op.so",
    srcs = [
        "kernels/repeat_op.cc",
        "kernels/repeat_op.h",
        "kernels/repeat_op_cpu_impl.cc",
        "kernels/repeat_op_gpu_impl.cc",
        "ops/repeat_op.cc",
    ],
    gpu_srcs = [
        "kernels/repeat_op_gpu_impl.cu.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["repeat_op"],
)

tf_gen_op_wrapper_py(
    name = "repeat_op",
    deps = [":repeat_op_op_lib"],
)

py_library(
    name = "repeat_py",
    srcs = [
        "__init__.py",
        "python/ops/repeat_op.py",
    ],
    data = [
        ":python/ops/_repeat_op.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":repeat_op",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "repeat_op_test",
    size = "small",
    srcs = ["python/kernel_tests/repeat_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":repeat_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
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
