# Description:
#   Contains the Keras initializer API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = [
        "//tensorflow/python/keras/google/private_tf_api_test:__pkg__",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "__init__.py",
        "initializers_v1.py",
        "initializers_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_inspect",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [
        ":initializers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/keras/engine",
        "//third_party/py/numpy",
    ],
)
