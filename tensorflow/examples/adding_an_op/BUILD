# Description:
# Code examples referenced by adding_an_op

load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_custom_op_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_1.so",
    srcs = ["zero_out_op_kernel_1.cc"],
)

py_library(
    name = "zero_out_op_1",
    srcs = ["zero_out_op_1.py"],
    data = [":zero_out_op_kernel_1.so"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "zero_out_grad_2",
    srcs = ["zero_out_grad_2.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        ":zero_out_op_1",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "zero_out_2_test",
    size = "small",
    srcs = ["zero_out_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notap",
    ],
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
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notap",
    ],
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
    srcs_version = "PY3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "cuda_op_test",
    size = "small",
    srcs = ["cuda_op_test.py"],
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
    python_version = "PY3",
    srcs_version = "PY3",
    tags = tf_cuda_tests_tags() + [
        "notap",
        "no_pip",
    ],
    deps = [
        ":cuda_op",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "fact_test",
    size = "small",
    srcs = ["fact_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow:tensorflow_py"],
)

tf_cc_binary(
    name = "attr_examples",
    srcs = ["attr_examples.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
    ],
)
