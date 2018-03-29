licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

package(default_visibility = ["//visibility:public"])

tf_proto_library(
    name = "fertile_stats_proto",
    srcs = ["fertile_stats.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/contrib/decision_trees/proto:generic_tree_model"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "tensor_forest_params_proto",
    srcs = ["tensor_forest_params.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/contrib/decision_trees/proto:generic_tree_model"],
    visibility = ["//visibility:public"],
)
