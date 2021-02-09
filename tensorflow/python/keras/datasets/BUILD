# Description:
#   Contains the Keras datasets package (internal TensorFlow version).

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "datasets",
    srcs = [
        "__init__.py",
        "boston_housing.py",
        "cifar.py",
        "cifar10.py",
        "cifar100.py",
        "fashion_mnist.py",
        "imdb.py",
        "mnist.py",
        "reuters.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:engine_utils",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
