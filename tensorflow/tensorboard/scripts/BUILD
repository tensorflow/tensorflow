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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/tensorboard/backend:server",
    ],
)

py_binary(
    name = "generate_testdata",
    srcs = ["generate_testdata.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__subpackages__"],
)
