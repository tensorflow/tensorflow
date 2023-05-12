load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud", "tsl_grpc_cc_dependencies")
load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
        "//tensorflow/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "preemption_notifier_test",
    size = "small",
    srcs = ["preemption_notifier_test.cc"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "preemption_sync_manager",
    srcs = ["preemption_sync_manager.cc"],
    hdrs = ["preemption_sync_manager.h"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/lib/monitoring:gauge",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "preemption_sync_manager_test",
    size = "small",
    srcs = ["preemption_sync_manager_test.cc"],
    deps = [
        ":preemption_notifier",
        ":preemption_sync_manager",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_client",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_impl",
        "//tensorflow/tsl/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tensorflow/tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc_impl",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc_impl",
        "//tensorflow/tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ] + tsl_grpc_cc_dependencies(),
)
