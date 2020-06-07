# Description:
#   Contains stateless ops for Keras layers.

package(
    default_visibility = [
        "//tensorflow/python/keras/layers:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
    ],
)
