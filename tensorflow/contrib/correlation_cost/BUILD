# Description:
#   An implementation of the correlation layer computing a cost volume

package(
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/...",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_py_library(
    name = "correlation_cost_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [":python/ops/_correlation_cost_op.so"],
    kernels = [
        ":correlation_cost_op_kernels",
        ":correlation_cost_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":correlation_cost_op",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ]
)

tf_kernel_library(
    name = "correlation_cost_op_kernels",
    srcs = [
        "kernels/correlation_cost_op.cc",
        "kernels/correlation_cost_op.h",
        "ops/correlation_cost_op.cc",
    ],
    gpu_srcs = [
        "kernels/correlation_cost_op_gpu.cu.cc",
    ],
    prefix = "correlation_cost_op",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/kernels:gpu_util_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "//third_party/eigen3"
    ] + if_cuda([
        "@cub_archive//:cub",
        "//tensorflow/core:gpu_runtime",
    ]),
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_correlation_cost_op.so",
    srcs = [
        "kernels/correlation_cost_op.cc",
        "kernels/correlation_cost_op.h",
        "ops/correlation_cost_op.cc",
    ],
    gpu_srcs = [
        "kernels/correlation_cost_op_gpu.cu.cc",
    ],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/kernels:gpu_util_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs"
    ] + if_cuda([
        "@cub_archive//:cub"
    ])
)

tf_gen_op_libs(
    op_lib_names = ["correlation_cost_op"],
    deps = ["//tensorflow/core:lib_proto_parsing"],
)

tf_gen_op_wrapper_py(
    name = "correlation_cost_op",
    deps = [":correlation_cost_op_op_lib"],
)

cuda_py_test(
    name = "correlation_cost_op_test",
    size = "medium",
    srcs = ["python/kernel_tests/correlation_cost_op_test.py"],
    additional_deps = [
        ":correlation_cost_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "//tensorflow/python:constant_op",
    ],
)
