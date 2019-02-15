package(
    default_visibility = [
        "//engedu/ml/tf_from_scratch:__pkg__",
        "//tensorflow:internal",
    ],
)

load("//tensorflow:tensorflow.bzl", "py_test")

licenses(["notice"])  # Apache 2.0

py_library(
    name = "losses",
    srcs = [
        "__init__.py",
        "losses.py",
        "losses_impl.py",
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)
