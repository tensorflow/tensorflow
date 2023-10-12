load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl:tsl.default.bzl", "get_compatible_with_portable", "tsl_grpc_cc_dependencies")
load("//tsl:tsl.bzl", "set_external_visibility")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "preemption_notifier_test",
    size = "small",
    srcs = ["preemption_notifier_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":preemption_notifier",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "preemption_sync_manager",
    srcs = ["preemption_sync_manager.cc"],
    hdrs = ["preemption_sync_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":preemption_notifier",
        "//tsl/distributed_runtime:call_options",
        "//tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tsl/lib/monitoring:gauge",
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "preemption_sync_manager_test",
    size = "small",
    srcs = ["preemption_sync_manager_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":preemption_notifier",
        ":preemption_sync_manager",
        "//tsl/distributed_runtime/coordination:coordination_client",
        "//tsl/distributed_runtime/coordination:coordination_service",
        "//tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tsl/distributed_runtime/coordination:coordination_service_impl",
        "//tsl/distributed_runtime/rpc:async_service_interface",
        "//tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:coordination_config_proto_cc_impl",
        "//tsl/protobuf:coordination_service_proto_cc_impl",
        "//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ] + tsl_grpc_cc_dependencies(),
)
