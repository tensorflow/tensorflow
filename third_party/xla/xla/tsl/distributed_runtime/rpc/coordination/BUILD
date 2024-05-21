load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "grpc_coordination_client",
    srcs = ["grpc_coordination_client.cc"],
    hdrs = ["grpc_coordination_client.h"],
    deps = [
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_client",
        "//xla/tsl/distributed_runtime/rpc:grpc_channel",
        "//xla/tsl/distributed_runtime/rpc:grpc_client_cq_tag",
        "//xla/tsl/distributed_runtime/rpc:grpc_state",
        "//xla/tsl/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_coordination_service_impl",
    srcs = ["grpc_coordination_service_impl.cc"],
    hdrs = ["grpc_coordination_service_impl.h"],
    deps = [
        "//xla/tsl/distributed_runtime/coordination:coordination_service",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_rpc_handler",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc:grpc_call",
        "//xla/tsl/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/protobuf:coordination_service_cc_grpc_proto",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)
