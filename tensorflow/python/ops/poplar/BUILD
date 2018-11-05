package(
    default_visibility = ["//tensorflow:internal"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

# Using same licence as the poplar plugin
licenses(["restricted"])

py_library(
    name = "poplar",
    srcs_version = "PY2AND3",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public",],
    deps = [
        ":popnn_ops",
        ":popnn_ops_grad",
    ],
)

py_library(
    name = "popnn_ops",
    srcs = ["popnn_ops.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public",],
    deps = [
        "//tensorflow/compiler/plugin/poplar:popnn_ops_wrapper_py",
    ],
)

py_library(
    name = "popnn_ops_grad",
    srcs = ["popnn_ops_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_wrapper_py",
    ],
)
