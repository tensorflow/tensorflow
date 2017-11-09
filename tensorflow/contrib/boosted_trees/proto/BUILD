licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_proto_library(
    name = "learner_proto",
    srcs = [
        "learner.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "quantiles_proto",
    srcs = [
        "quantiles.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "split_info_proto",
    srcs = ["split_info.proto"],
    cc_api_version = 2,
    protodeps = [
        ":tree_config_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "tree_config_proto",
    srcs = ["tree_config.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
