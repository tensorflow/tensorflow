# Description:
#   Contains the Keras wrapper API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "wrappers",
    srcs = [
        "__init__.py",
        "scikit_learn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras/utils:generic_utils",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "scikit_learn_test",
    size = "small",
    srcs = ["scikit_learn_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":wrappers",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)
