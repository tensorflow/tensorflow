# Description:
# Various tools and rules related to the TensorFlow docker container.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)
