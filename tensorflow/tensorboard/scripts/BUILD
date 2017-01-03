# Description:
# Some useful scripts that are bundled with TensorBoard.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "serialize_tensorboard",
    srcs = ["serialize_tensorboard.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/tensorboard/backend:server",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "generate_testdata",
    srcs = ["generate_testdata.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__subpackages__"],
)
