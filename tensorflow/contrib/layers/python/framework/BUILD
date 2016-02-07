# Description:
#   contains TensorFlow framework that is experimental or unstable and not yet supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

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

py_library(
    name = "tensor_util",
    srcs = ["tensor_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "tensor_util_test",
    size = "small",
    srcs = ["tensor_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
    ],
)
