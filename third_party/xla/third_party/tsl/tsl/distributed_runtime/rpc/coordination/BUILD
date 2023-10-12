load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("//tsl:tsl.bzl", "set_external_visibility")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_coordination_client",
    srcs = ["grpc_coordination_client.cc"],
    hdrs = ["grpc_coordination_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/distributed_runtime:call_options",
        "//tsl/distributed_runtime/coordination:coordination_client",
        "//tsl/distributed_runtime/rpc:grpc_channel",
        "//tsl/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tsl/distributed_runtime/rpc:grpc_state",
        "//tsl/distributed_runtime/rpc:grpc_util",
        "//tsl/platform:mutex",
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "grpc_coordination_service_impl",
    srcs = ["grpc_coordination_service_impl.cc"],
    hdrs = ["grpc_coordination_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tsl/distributed_runtime/coordination:coordination_service_rpc_handler",
        "//tsl/distributed_runtime/rpc:async_service_interface",
        "//tsl/distributed_runtime/rpc:grpc_call",
        "//tsl/distributed_runtime/rpc:grpc_util",
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/protobuf:coordination_service_cc_grpc_proto",
        "//tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)
