# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "sdca_kernel",
    srcs = ["kernels/sdca_ops.cc"],
    hdrs = ["kernels/logistic-loss.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sdca_ops",
    srcs = ["ops/sdca_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["sdca_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_sdca_ops_py",
    out = "gen_sdca_ops.py",
    deps = [":sdca_ops"],
)

py_library(
    name = "sdca_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/sdca_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_sdca_ops_py",
    ],
)

py_test(
    name = "sdca_ops_test",
    srcs = ["python/kernel_tests/sdca_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_kernel",
        ":sdca_ops",
        ":sdca_ops_py",
        "//third_party/py/tensorflow",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:protos_all_py_pb2",
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
