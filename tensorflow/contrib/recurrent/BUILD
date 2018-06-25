# Recurrent library.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "gpu_py_tests")

py_library(
    name = "recurrent_py",
    srcs = ["python/recurrent_api.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":functional_rnn_ops_py",
        ":recurrent_ops_py",
    ],
)

py_library(
    name = "recurrent_ops_py",
    srcs = ["python/ops/recurrent.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "functional_rnn_ops_py",
    srcs = ["python/ops/functional_rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent_ops_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:standard_ops",
    ],
)

gpu_py_tests(
    name = "recurrent_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/recurrent_test.py"],
    additional_deps = [
        ":recurrent_ops_py",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:variables",
    ],
    tags = ["nopip"],
)

gpu_py_tests(
    name = "functional_rnn_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/functional_rnn_test.py"],
    additional_deps = [
        ":functional_rnn_ops_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/tpu:tpu",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "nopip",
        "optonly",
    ],
)
