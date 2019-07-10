# Description:
#   Contains the Keras Premade Models (internal TensorFlow version).

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "premade",
    srcs = [
        "linear.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:backend_config",
        "//tensorflow/python/keras:regularizers",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["linear_test.py"],
    shard_count = 2,
    deps = [
        ":premade",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)
