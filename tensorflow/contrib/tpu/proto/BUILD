licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

tf_proto_library(
    name = "tpu_embedding_config_proto",
    srcs = [
        "tpu_embedding_config.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "topology_proto",
    srcs = [
        "topology.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
