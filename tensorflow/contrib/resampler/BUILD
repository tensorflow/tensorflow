licenses(["notice"])  # Apache 2.0 License

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

tf_custom_op_py_library(
    name = "resampler_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [":python/ops/_resampler_ops.so"],
    kernels = [
        ":resampler_ops_kernels",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":resampler_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

tf_kernel_library(
    name = "resampler_ops_kernels",
    srcs = [
        "kernels/resampler_ops.cc",
        "kernels/resampler_ops.h",
    ],
    gpu_srcs = [
        "kernels/resampler_ops_gpu.cu.cc",
        "kernels/resampler_ops.h",
    ],
    prefix = "resampler_ops",
    deps = [
        ":resampler_ops_op_lib",
        "//tensorflow/compiler/tf2xla/kernels:resampler_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_resampler_ops.so",
    srcs = [
        "kernels/resampler_ops.cc",
        "kernels/resampler_ops.h",
        "ops/resampler_ops.cc",
    ],
    gpu_srcs = [
        "kernels/resampler_ops_gpu.cu.cc",
        "kernels/resampler_ops.h",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "resampler_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "resampler_ops",
    deps = [":resampler_ops_op_lib"],
)

cuda_py_test(
    name = "resampler_ops_test",
    size = "small",
    srcs = ["python/ops/resampler_ops_test.py"],
    additional_deps = [
        ":resampler_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:array_ops",
    ],
)
