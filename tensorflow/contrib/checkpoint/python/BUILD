licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "split_dependency",
    srcs = ["split_dependency.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:training",
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
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
    ],
)
