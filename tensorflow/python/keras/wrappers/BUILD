# Description:
#   Contains the Keras wrapper API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "wrappers",
    srcs = [
        "__init__.py",
        "scikit_learn.py",
    ],
    srcs_version = "PY3",
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
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/utils:np_utils",
        "//third_party/py/numpy",
    ],
)
