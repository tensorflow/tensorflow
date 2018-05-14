package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

py_library(
    name = "rpc_op_py",
    srcs = ["rpc_op.py"],
    deps = [
        ":gen_rpc_op_py",
        "//tensorflow/python:framework_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_rpc_op_py",
    out = "gen_rpc_op.py",
    deps = [
        "//tensorflow/core:rpc_ops_op_lib",
    ],
)
