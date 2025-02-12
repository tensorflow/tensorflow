# Description:
# Code examples referenced by adding_an_op

load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_custom_op_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_1.so",
    srcs = ["zero_out_op_kernel_1.cc"],
)

py_strict_library(
    name = "zero_out_op_1",
    srcs = ["zero_out_op_1.py"],
    data = [":zero_out_op_kernel_1.so"],
    deps = ["//tensorflow:tensorflow_py"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_2.so",
    srcs = ["zero_out_op_kernel_2.cc"],
)

py_strict_library(
    name = "zero_out_op_2",
    srcs = ["zero_out_op_2.py"],
    data = [":zero_out_op_kernel_2.so"],
    deps = ["//tensorflow:tensorflow_py"],
)

tf_custom_op_library(
    name = "zero_out_op_kernel_3.so",
    srcs = ["zero_out_op_kernel_3.cc"],
)

py_strict_library(
    name = "zero_out_op_3",
    srcs = ["zero_out_op_3.py"],
    data = [":zero_out_op_kernel_3.so"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_strict_library(
    name = "zero_out_grad_2",
    srcs = ["zero_out_grad_2.py"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:sparse_ops",
    ],
)

py_strict_test(
    name = "zero_out_1_test",
    size = "medium",
    srcs = ["zero_out_1_test.py"],
    tags = [
        "notap",
    ],
    deps = [
        ":zero_out_op_1",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_test(
    name = "zero_out_2_test",
    size = "medium",
    srcs = ["zero_out_2_test.py"],
    tags = [
        "notap",
    ],
    deps = [
        ":zero_out_grad_2",
        ":zero_out_op_2",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_test(
    name = "zero_out_3_test",
    size = "medium",
    srcs = ["zero_out_3_test.py"],
    tags = [
        "notap",
    ],
    deps = [
        ":zero_out_op_3",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
    ],
)

tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)

py_strict_library(
    name = "cuda_op",
    srcs = ["cuda_op.py"],
    data = [":cuda_op_kernel.so"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_strict_test(
    name = "cuda_op_test",
    size = "small",
    srcs = ["cuda_op_test.py"],
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
    tags = tf_cuda_tests_tags() + [
        "notap",
    ],
    deps = [
        ":cuda_op",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_test(
    name = "fact_test",
    size = "small",
    srcs = ["fact_test.py"],
    deps = [
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:test_lib",
    ],
)

tf_cc_binary(
    name = "attr_examples",
    srcs = ["attr_examples.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
    ],
)
