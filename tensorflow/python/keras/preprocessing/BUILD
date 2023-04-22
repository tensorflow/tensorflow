# Description:
#   Contains the Keras preprocessing layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":image",
        ":sequence",
        ":text",
        ":timeseries",
        "//tensorflow/python/keras/utils:all_utils",
    ],
)

py_library(
    name = "image",
    srcs = [
        "dataset_utils.py",
        "image.py",
        "image_dataset.py",
    ],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras/utils:data_utils",
    ],
)

py_library(
    name = "timeseries",
    srcs = [
        "timeseries.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "text",
    srcs = [
        "dataset_utils.py",
        "text.py",
        "text_dataset.py",
    ],
    srcs_version = "PY3",
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "image_dataset_test",
    size = "small",
    srcs = ["image_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":image",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras",
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

tf_py_test(
    name = "text_dataset_test",
    size = "small",
    srcs = ["text_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":text",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "timeseries_test",
    size = "small",
    srcs = ["timeseries_test.py"],
    python_version = "PY3",
    deps = [
        ":timeseries",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//third_party/py/numpy",
    ],
)
