licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "checkpoint",
    srcs_version = "PY2AND3",
    deps = [
        ":containers",
        ":python_state",
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

tf_py_test(
    name = "containers_test",
    srcs = ["containers_test.py"],
    additional_deps = [
        ":containers",
        "@six_archive//:six",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/training/checkpointable:base",
        "//tensorflow/python/training/checkpointable:util",
    ],
)

py_library(
    name = "python_state",
    srcs = ["python_state.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/training/checkpointable:base",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "python_state_test",
    srcs = ["python_state_test.py"],
    additional_deps = [
        ":python_state",
        "//third_party/py/numpy",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training/checkpointable:util",
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

tf_py_test(
    name = "split_dependency_test",
    srcs = ["split_dependency_test.py"],
    additional_deps = [
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

tf_py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    additional_deps = [
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
