# Description:
#   Contains the Keras datasets package (internal TensorFlow version).

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

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
    srcs_version = "PY2AND3",
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
