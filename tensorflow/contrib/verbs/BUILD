# Description:
#   Verbs RDMA communication interfaces and implementations for TensorFlow.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

tf_proto_library_cc(
    name = "verbs_service_proto",
    srcs = ["verbs_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

cc_library(
    name = "verbs_util",
    srcs = ["verbs_util.cc"],
    hdrs = ["verbs_util.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "grpc_verbs_service",
    srcs = ["grpc_verbs_service.cc"],
    hdrs = ["grpc_verbs_service.h"],
    deps = [
        ":grpc_verbs_service_impl",
        ":rdma_mgr",
        ":verbs_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@grpc//:grpc++_unsecure",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_verbs_service_impl",
    srcs = ["grpc_verbs_service_impl.cc"],
    hdrs = ["grpc_verbs_service_impl.h"],
    deps = [
        ":verbs_service_proto_cc",
        "@grpc//:grpc++_unsecure",
    ],
)

cc_library(
    name = "grpc_verbs_client",
    srcs = ["grpc_verbs_client.cc"],
    hdrs = ["grpc_verbs_client.h"],
    deps = [
        ":grpc_verbs_service_impl",
        ":verbs_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rdma_rendezvous_mgr",
    srcs = ["rdma_rendezvous_mgr.cc"],
    hdrs = ["rdma_rendezvous_mgr.h"],
    deps = [
        ":rdma_mgr",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
)

cc_library(
    name = "rdma_mgr",
    srcs = ["rdma_mgr.cc"],
    hdrs = ["rdma_mgr.h"],
    deps = [
        ":grpc_verbs_client",
        ":rdma",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
    ],
)

cc_library(
    name = "rdma",
    srcs = ["rdma.cc"],
    hdrs = ["rdma.h"],
    linkopts = select({
        "//tensorflow:with_verbs_support": ["-libverbs"],
        "//conditions:default": [],
    }),
    deps = [
        ":verbs_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
)

cc_library(
    name = "verbs_server_lib",
    srcs = ["verbs_server_lib.cc"],
    hdrs = ["verbs_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":grpc_verbs_service",
        ":rdma_mgr",
        ":rdma_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)
