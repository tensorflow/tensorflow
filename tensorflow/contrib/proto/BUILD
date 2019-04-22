package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "proto",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
        "//tensorflow/contrib/proto/python/ops:encode_proto_op_py",
        "//tensorflow/python:proto_ops",
    ],
)

tf_py_test(
    name = "import_test",
    srcs = ["import_test.py"],
    additional_deps = [
        ":proto",
        "//tensorflow/python:client_testlib",
    ],
)
