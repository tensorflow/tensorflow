# Description:
#   Contains the Keras Premade Models (internal TensorFlow version).
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "premade",
    srcs = [
        "__init__.py",
        "linear.py",
        "wide_deep.py",
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
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":premade",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "wide_deep_test",
    size = "medium",
    srcs = ["wide_deep_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":premade",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)
