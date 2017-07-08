licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_reduce_slice_ops.so",
    srcs = [
        "kernels/reduce_slice_ops.cc",
        "ops/reduce_slice_ops.cc",
    ],
    gpu_srcs = [
        "kernels/reduce_slice_ops_gpu.cu.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["reduce_slice_ops"],
)

tf_gen_op_wrapper_py(
    name = "reduce_slice_ops",
    deps = [":reduce_slice_ops_op_lib"],
)

py_library(
    name = "reduce_slice_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/reduce_slice_ops.py",
    ],
    data = [
        ":python/ops/_reduce_slice_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":reduce_slice_ops"
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "reduce_slice_ops_test",
    size = "small",
    srcs = ["python/ops/reduce_slice_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":reduce_slice_ops_py",
        "//tensorflow/python:platform_test_util",
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
