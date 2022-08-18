load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_coordination_client",
    srcs = ["grpc_coordination_client.cc"],
    hdrs = ["grpc_coordination_client.h"],
    deps = [
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/coordination:coordination_client",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/core/distributed_runtime/rpc:grpc_state",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_coordination_service_impl",
    srcs = ["grpc_coordination_service_impl.cc"],
    hdrs = ["grpc_coordination_service_impl.h"],
    deps = [
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_cc_grpc_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core:platform_base",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_rpc_handler",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/platform",
        "//tensorflow/core/distributed_runtime:worker_env",
    ] + tf_grpc_cc_dependencies(),
)
