package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
)

py_library(
    name = "decode_proto_op_py",
    srcs = ["decode_proto_op.py"],
    deps = [
        ":gen_decode_proto_op_py",
        "//tensorflow/python:framework_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_decode_proto_op_py",
    out = "gen_decode_proto_op.py",
    deps = [
        "//tensorflow/core:decode_proto_ops_op_lib",
    ],
)

py_library(
    name = "encode_proto_op_py",
    srcs = ["encode_proto_op.py"],
    deps = [
        ":gen_encode_proto_op_py",
        "//tensorflow/python:framework_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_encode_proto_op_py",
    out = "gen_encode_proto_op.py",
    deps = [
        "//tensorflow/core:encode_proto_ops_op_lib",
    ],
)
