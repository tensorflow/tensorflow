licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_kernel_tests_linkstatic")

tf_custom_op_library(
    name = "python/ops/_reduce_slice_ops.so",
    srcs = [
        "kernels/reduce_slice_ops.cc",
        "kernels/reduce_slice_ops.h",
        "ops/reduce_slice_ops.cc",
    ],
    gpu_srcs = [
        "kernels/reduce_slice_ops.h",
        "kernels/reduce_slice_ops_gpu.cu.cc",
    ],
)

tf_kernel_library(
    name = "reduce_slice_ops_kernels",
    srcs = [
        "kernels/reduce_slice_ops.cc",
    ],
    hdrs = [
        "kernels/reduce_slice_ops.h",
    ],
    gpu_srcs = [
        "kernels/reduce_slice_ops.h",
        "kernels/reduce_slice_ops_gpu.cu.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["reduce_slice_ops"],
)

tf_gen_op_wrapper_py(
    name = "reduce_slice_ops",
    deps = [":reduce_slice_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "reduce_slice_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/reduce_slice_ops.py",
    ],
    dso = [
        ":python/ops/_reduce_slice_ops.so",
    ],
    kernels = [
        ":reduce_slice_ops_kernels",
        ":reduce_slice_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":reduce_slice_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "reduce_slice_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/reduce_slice_ops_test.py"],
    additional_deps = [
        ":reduce_slice_ops_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_cc_test(
    name = "reduce_slice_ops_test_cc",
    size = "small",
    srcs = [
        "ops/reduce_slice_ops_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":reduce_slice_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
