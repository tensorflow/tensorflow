# Description:
# A plugin system for TensorBoard

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "base_plugin",
    srcs = ["base_plugin.py"],
    srcs_version = "PY2AND3",
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    visibility = ["//tensorflow:__pkg__"],
)
