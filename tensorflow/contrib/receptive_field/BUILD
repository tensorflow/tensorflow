# Description:
#   Contains modules to compute receptive field parameters for CNN models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "receptive_field_pip",
    deps = [
        ":receptive_field_py",
    ],
)

py_library(
    name = "graph_compute_order_py",
    srcs = [
        "python/util/graph_compute_order.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":parse_layer_parameters_py",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "parse_layer_parameters_py",
    srcs = [
        "python/util/parse_layer_parameters.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "receptive_field_py",
    srcs = [
        "python/util/parse_layer_parameters.py",
        "python/util/receptive_field.py",
        "receptive_field_api.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_compute_order_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_compute_order_test",
    srcs = ["python/util/graph_compute_order_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_compute_order_py",
        ":receptive_field_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
    ],
)

py_test(
    name = "parse_layer_parameters_test",
    srcs = ["python/util/parse_layer_parameters_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_compute_order_py",
        ":parse_layer_parameters_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
    ],
)

py_test(
    name = "receptive_field_test",
    srcs = ["python/util/receptive_field_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":receptive_field_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
        "//third_party/py/numpy",
    ],
)
