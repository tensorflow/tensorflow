# Description:
# Transfer learning example for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "retrain",
    srcs = [
        "retrain.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "retrain_test",
    size = "small",
    srcs = [
        "retrain.py",
        "retrain_test.py",
    ],
    data = [
        ":data/labels.txt",
        "//tensorflow/examples/label_image:data/grace_hopper.jpg",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":retrain",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)
