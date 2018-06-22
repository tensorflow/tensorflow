# Description:
#   Utilities for reading and writing object-based checkpoints.

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:platform",
        "//tensorflow/python:saveable_object",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "tracking",
    srcs = ["tracking.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
    ],
)

py_test(
    name = "tracking_test",
    srcs = ["tracking_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":tracking",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "data_structures_base",
    srcs = ["data_structures_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
    ],
)

py_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":data_structures_base",
    ],
)

py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_structures",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":tracking",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:ops",
        "//tensorflow/python:saveable_object",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
        "notsan",  # b/74395663
    ],
    deps = [
        ":base",
        ":util",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:training_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "@six_archive//:six",
    ],
)
