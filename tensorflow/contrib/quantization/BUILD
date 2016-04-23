# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_custom_op_library",
)

cc_library(
    name = "cc_array_ops",
    srcs = ["ops/array_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cc_math_ops",
    srcs = ["ops/math_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cc_nn_ops",
    srcs = ["ops/nn_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cc_ops",
    linkstatic = 1,
    deps = [
        ":cc_array_ops",
        ":cc_math_ops",
        ":cc_nn_ops",
    ],
    alwayslink = 1,
)

filegroup(
    name = "android_ops",
    srcs = glob(["ops/*.cc"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "quantization_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "python/array_ops.py",
        "python/math_ops.py",
        "python/nn_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "array_ops",
    deps = ["//tensorflow/contrib/quantization:cc_array_ops"],
)

tf_gen_op_wrapper_py(
    name = "math_ops",
    deps = ["//tensorflow/contrib/quantization:cc_math_ops"],
)

tf_gen_op_wrapper_py(
    name = "nn_ops",
    deps = ["//tensorflow/contrib/quantization:cc_nn_ops"],
)

py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["python/dequantize_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/quantization:quantized_ops_py",
        "//tensorflow/contrib/quantization/kernels:quantized_kernels_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["python/quantized_conv_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/quantization:quantized_ops_py",
        "//tensorflow/contrib/quantization/kernels:quantized_kernels_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)

tf_custom_op_library(
    name = "_quantized_ops.so",
    srcs = [
        "ops/array_ops.cc",
        "ops/math_ops.cc",
        "ops/nn_ops.cc",
    ],
    deps = [
    ],
)

py_library(
    name = "quantized_ops_py",
    srcs = ["load_quantized_ops_so.py"],
    data = ["_quantized_ops.so"],
    srcs_version = "PY2AND3",
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
