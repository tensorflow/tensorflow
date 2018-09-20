# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

# TODO(yleon): Refactor after one we switching to the V2 kernels.
py_library(
    name = "lookup_py",
    srcs = [
        "__init__.py",
        "lookup_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:lookup_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "lookup_ops_test",
    size = "medium",
    srcs = ["lookup_ops_test.py"],
    additional_deps = [
        ":lookup_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
    grpc_enabled = True,
    tags = ["no_windows"],  # TODO: needs investigation on Windows
)
