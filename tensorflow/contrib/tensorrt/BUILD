# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
    "tf_custom_op_library_additional_deps",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)

tf_cuda_library(
    name = "trt_shape_function",
    srcs = ["shape_fn/trt_shfn.cc"],
    hdrs = ["shape_fn/trt_shfn.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2tensorrt:trt_logging",
        "//tensorflow/compiler/tf2tensorrt:trt_plugins",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]) + tf_custom_op_library_additional_deps(),
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/trt_convert.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/compiler/tensorrt:init_py",
    ],
)

# The following rules forward the libraries that were moved in order to not
# break other internal targets.

alias(
    name = "trt_conversion",
    actual = "//tensorflow/compiler/tf2tensorrt:trt_conversion",
)

alias(
    name = "trt_op_kernels",
    actual = "//tensorflow/compiler/tf2tensorrt:trt_op_kernels",
)

alias(
    name = "trt_engine_op_op_lib",
    actual = "//tensorflow/compiler/tf2tensorrt:trt_engine_op_op_lib",
)
