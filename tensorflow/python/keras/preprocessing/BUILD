# Description:
#   Contains the Keras preprocessing layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":image",
        ":sequence",
        ":text",
    ],
)

py_library(
    name = "image",
    srcs = [
        "image.py",
    ],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:data_utils",
    ],
)

py_library(
    name = "sequence",
    srcs = [
        "sequence.py",
    ],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras/utils:data_utils",
    ],
)

py_library(
    name = "text",
    srcs = [
        "text.py",
    ],
    deps = ["//tensorflow/python:util"],
)

tf_py_test(
    name = "image_test",
    size = "medium",
    srcs = ["image_test.py"],
    python_version = "PY3",
    deps = [
        ":image",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sequence_test",
    size = "small",
    srcs = ["sequence_test.py"],
    python_version = "PY3",
    deps = [
        ":sequence",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    python_version = "PY3",
    deps = [
        ":text",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)
