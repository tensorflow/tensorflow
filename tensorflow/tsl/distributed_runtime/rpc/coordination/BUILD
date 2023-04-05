load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/tsl:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_coordination_client",
    srcs = ["grpc_coordination_client.cc"],
    hdrs = ["grpc_coordination_client.h"],
    deps = [
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_client",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_state",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "grpc_coordination_service_impl",
    srcs = ["grpc_coordination_service_impl.cc"],
    hdrs = ["grpc_coordination_service_impl.h"],
    deps = [
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_rpc_handler",
        "//tensorflow/tsl/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_call",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/protobuf:coordination_service_cc_grpc_proto",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)
