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
    name = "python/ops/_libsvm_ops.so",
    srcs = [
        "kernels/decode_libsvm_op.cc",
        "ops/libsvm_ops.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_kernel_library(
    name = "libsvm_kernels",
    srcs = ["kernels/decode_libsvm_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["libsvm_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "libsvm_ops",
    deps = [":libsvm_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "libsvm",
    srcs = [
        "__init__.py",
        "python/ops/libsvm_ops.py",
    ],
    dso = [
        ":python/ops/_libsvm_ops.so",
    ],
    kernels = [
        ":libsvm_kernels",
        ":libsvm_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":libsvm_ops",
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
    name = "decode_libsvm_op_test",
    srcs = ["python/kernel_tests/decode_libsvm_op_test.py"],
    additional_deps = [
        ":libsvm",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
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
