# Description:
# Code examples referenced by adding_an_op

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cuda_tests_tags")

exports_files(["LICENSE"])

tf_custom_op_library(
    name = "zero_out_op_kernel_1.so",
    srcs = ["zero_out_op_kernel_1.cc"],
)

py_library(
    name = "zero_out_op_1",
    srcs = ["zero_out_op_1.py"],
    data = [":zero_out_op_kernel_1.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_2.so",
    srcs = ["zero_out_op_kernel_2.cc"],
)

py_library(
    name = "zero_out_op_2",
    srcs = ["zero_out_op_2.py"],
    data = [":zero_out_op_kernel_2.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_3.so",
    srcs = ["zero_out_op_kernel_3.cc"],
)

py_library(
    name = "zero_out_op_3",
    srcs = ["zero_out_op_3.py"],
    data = [":zero_out_op_kernel_3.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "zero_out_grad_2",
    srcs = ["zero_out_grad_2.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

py_test(
    name = "zero_out_1_test",
    size = "small",
    srcs = ["zero_out_1_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":zero_out_op_1",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "zero_out_2_test",
    size = "small",
    srcs = ["zero_out_2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":zero_out_grad_2",
        ":zero_out_op_2",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "zero_out_3_test",
    size = "small",
    srcs = ["zero_out_3_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":zero_out_op_3",
        "//tensorflow:tensorflow_py",
    ],
)

tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)

py_library(
    name = "cuda_op",
    srcs = ["cuda_op.py"],
    data = [":cuda_op_kernel.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "cuda_op_test",
    size = "small",
    srcs = ["cuda_op_test.py"],
    srcs_version = "PY2AND3",
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cuda_op",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "fact_test",
    size = "small",
    srcs = ["fact_test.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

cc_binary(
    name = "attr_examples",
    srcs = ["attr_examples.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
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
