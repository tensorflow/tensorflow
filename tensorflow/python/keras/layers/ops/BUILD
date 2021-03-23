# Description:
#   Contains stateless ops for Keras layers.

package(
    default_visibility = [
        "//tensorflow/python/keras/layers:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
    ],
)
