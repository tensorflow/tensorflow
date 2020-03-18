load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

tf_proto_library(
    name = "master_proto",
    srcs = ["master.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

tf_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

cc_grpc_library(
    name = "master_cc_grpc_proto",
    srcs = [":master_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":master_proto_cc"],
)

cc_grpc_library(
    name = "worker_cc_grpc_proto",
    srcs = [":worker_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":worker_proto_cc"],
)
