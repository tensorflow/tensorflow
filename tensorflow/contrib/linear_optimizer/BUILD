# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_binary(
    name = "python/ops/_sdca_ops.so",
    srcs = [
        "kernels/logistic-loss.h",
        "kernels/sdca_ops.cc",
        "kernels/squared-loss.h",
        "ops/sdca_ops.cc",
    ],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-Bsymbolic",
            "-lm",
        ],
        "//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

py_library(
    name = "sdca_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/sdca_ops.py",
    ],
    data = [":python/ops/_sdca_ops.so"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "sdca_ops_test",
    srcs = ["python/kernel_tests/sdca_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops_py",
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
