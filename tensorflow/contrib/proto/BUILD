package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "proto",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
        "//tensorflow/contrib/proto/python/ops:encode_proto_op_py",
    ],
)
