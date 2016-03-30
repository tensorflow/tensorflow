# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "framework_py",
    srcs = [
        "__init__.py",
        "python/framework/__init__.py",
        "python/framework/tensor_util.py",
        "python/ops/__init__.py",
        "python/ops/ops.py",
        "python/ops/variables.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "ops_test",
    srcs = glob(["python/ops/ops_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "tensor_util_test",
    srcs = glob(["python/framework/tensor_util_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "variables_test",
    srcs = glob(["python/ops/variables_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":framework_py",
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
