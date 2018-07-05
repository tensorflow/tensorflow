licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "checkpoint",
    srcs_version = "PY2AND3",
    deps = [
        ":containers",
        ":split_dependency",
        ":visualize",
        "//tensorflow/python/training/checkpointable:data_structures",
    ],
)

py_library(
    name = "containers",
    srcs = ["containers.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/training/checkpointable:base",
        "//tensorflow/python/training/checkpointable:data_structures",
    ],
)

py_test(
    name = "containers_test",
    srcs = ["containers_test.py"],
    deps = [
        ":containers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/training/checkpointable:base",
        "//tensorflow/python/training/checkpointable:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "split_dependency",
    srcs = ["split_dependency.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/training/checkpointable:base",
    ],
)

py_test(
    name = "split_dependency_test",
    srcs = ["split_dependency_test.py"],
    deps = [
        ":split_dependency",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training/checkpointable:base",
        "//tensorflow/python/training/checkpointable:util",
    ],
)

py_library(
    name = "visualize",
    srcs = ["visualize.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/training/checkpointable:base",
        "//tensorflow/python/training/checkpointable:util",
    ],
)

py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    deps = [
        ":visualize",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/training/checkpointable:util",
    ],
)
