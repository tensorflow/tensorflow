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
        "//tensorflow/tensorboard:tensorboard_server",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__subpackages__"],
)
