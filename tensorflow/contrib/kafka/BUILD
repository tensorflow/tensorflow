package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_custom_op_library(
    name = "python/ops/_kafka_ops.so",
    srcs = [
        "kernels/kafka_reader_ops.cc",
        "ops/kafka_ops.cc",
    ],
    deps = [
        "//tensorflow/core:reader_base",
        "//tensorflow/core/kernels:bounds_check_lib",
        "@kafka//:kafka",
    ],
)

tf_kernel_library(
    name = "kafka_kernels",
    srcs = ["kernels/kafka_reader_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:reader_base",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//third_party/eigen3",
        "@kafka//:kafka",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["kafka_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "kafka_ops",
    deps = [":kafka_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "kafka",
    srcs = [
        "__init__.py",
        "python/ops/kafka_reader_ops.py",
    ],
    dso = [
        ":python/ops/_kafka_ops.so",
    ],
    kernels = [
        ":kafka_kernels",
        ":kafka_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":kafka_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "kafka_test",
    srcs = ["python/kernel_tests/kafka_test.py"],
    additional_deps = [
        ":kafka",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
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
