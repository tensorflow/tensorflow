# New implementations of control flow ops

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "control_flow",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cond_v2",
    ],
)

py_library(
    name = "cond_v2",
    srcs = ["python/cond_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:c_api_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:function_def_to_graph",
        "//tensorflow/python:functional_ops_gen",
        "//tensorflow/python:gradients",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "cond_v2_test",
    size = "small",
    srcs = ["python/cond_v2_test.py"],
    additional_deps = [
        ":cond_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:training",
    ],
    grpc_enabled = True,
)
