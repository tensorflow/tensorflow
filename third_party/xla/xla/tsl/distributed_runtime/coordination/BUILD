load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility", "tsl_gpu_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

tsl_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    deps = [
        ":coordination_service_error_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    deps = [
        "//xla/tsl/distributed_runtime:call_options",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_service",
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_client",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
    ],
)

# Keeping the implementation as a separate build target.
# This is an alwayslink library for statically registering "standalone" implementation.
# Other implementations of the service will be provided in the future.
tsl_gpu_library(
    name = "coordination_service_impl",
    srcs = ["coordination_service.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "test_device_proto",
    testonly = 1,
    srcs = ["test_device.proto"],
    cc_api_version = 2,
)

tsl_cc_test(
    name = "coordination_service_test",
    srcs = ["coordination_service_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        ":coordination_service_impl",
        ":test_device_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

tsl_gpu_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    deps = [
        ":coordination_client",
        ":coordination_service_error_util",
        "//xla/tsl/distributed_runtime:call_options",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/framework:cancellation",
        "@local_tsl//tsl/lib/monitoring:gauge",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

tsl_cc_test(
    name = "coordination_service_agent_test",
    srcs = ["coordination_service_agent_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service_agent",
        "//xla/tsl/distributed_runtime:call_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc_impl",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc_impl",
    ],
)

cc_library(
    name = "coordination_service_rpc_handler",
    srcs = ["coordination_service_rpc_handler.cc"],
    hdrs = [
        "coordination_service_rpc_handler.h",
    ],
    deps = [
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_error_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ],
)

tsl_cc_test(
    name = "coordination_service_recoverable_job_test",
    srcs = ["coordination_service_recoverable_job_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_impl",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc_impl",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc_impl",
        "@local_tsl//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
)
