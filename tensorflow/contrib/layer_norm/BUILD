# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//learning/brain:__subpackages__",
    "//tensorflow:__subpackages__",
])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

tf_custom_op_library(
    name = "python/ops/_layer_norm_fused_op.so",
    srcs = [
        "kernels/layer_norm_fused_op.cc",
        "kernels/layer_norm_fused_op.h",
        "kernels/layer_norm_fused_grad_op.cc",
        "ops/layer_norm_fused_op.cc",
        ],
    gpu_srcs = [
        "kernels/layer_norm_fused_op_gpu.cu.cc",
        "kernels/layer_norm_fused_op.h",
        ]
)

tf_gen_op_libs(
    op_lib_names = ["layer_norm_fused_op"],
)

tf_gen_op_wrapper_py(
    name = "layer_norm_fused_op",
    deps = [":layer_norm_fused_op_op_lib"],
)

tf_kernel_library(
    name = "layer_norm_fused_op_kernel",
    srcs = [
        "kernels/layer_norm_fused_op.cc",
        "kernels/layer_norm_fused_op.h",
        "kernels/layer_norm_fused_grad_op.cc",
        ],
    gpu_srcs = [
        "kernels/layer_norm_fused_op.h",
        ],
    deps = [
        "//tensorflow/core:framework",
    ],
)

py_library(
    name = "layer_norm_py",
    srcs = [
        "__init__.py",
        "python/layers/__init__.py",
        "python/layers/layer_norm.py",
        "python/ops/layer_norm_fused_op.py",
    ],
    data = [
        ":python/ops/_layer_norm_fused_op.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":layer_norm_fused_op",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "layer_norm_fused_test",
    size = "small",
    srcs = ["python/layers/layer_norm_fused_test.py"],
    additional_deps = [
        ":layer_norm_py",
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
    visibility = ["//tensorflow:__subpackages__"],
)
