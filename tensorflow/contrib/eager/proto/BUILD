licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

tf_proto_library(
    name = "checkpointable_object_graph_proto",
    srcs = [
        "checkpointable_object_graph.proto",
    ],
    visibility = ["//tensorflow/contrib/eager/python:__subpackages__"],
)
