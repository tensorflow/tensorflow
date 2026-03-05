load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
        "//third_party/py/jax:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

xla_cc_test(
    name = "preemption_notifier_test",
    size = "small",
    srcs = ["preemption_notifier_test.cc"],
    deps = [
        ":preemption_notifier",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "preemption_sync_manager",
    srcs = ["preemption_sync_manager.cc"],
    hdrs = ["preemption_sync_manager.h"],
    deps = [
        ":preemption_notifier",
        "//xla/pjrt/distributed/coordination:coordination_service_agent",
        "//xla/pjrt/distributed/coordination:coordination_service_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/lib/monitoring:gauge",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "preemption_sync_manager_test",
    size = "small",
    srcs = ["preemption_sync_manager_test.cc"],
    deps = [
        ":preemption_notifier",
        ":preemption_sync_manager",
        "//xla/pjrt/distributed/coordination:coordination_client",
        "//xla/pjrt/distributed/coordination:coordination_service",
        "//xla/pjrt/distributed/coordination:coordination_service_agent",
        "//xla/pjrt/distributed/coordination:grpc_coordination_client",
        "//xla/pjrt/distributed/coordination:grpc_coordination_service_impl",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
