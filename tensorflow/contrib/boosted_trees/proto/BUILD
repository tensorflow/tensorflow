licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

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

cc_library(
    name = "proto_impl",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
    ],
)

tf_cc_binary(
    name = "libboostedtreeprotos.so",
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [":proto_impl"],
)
