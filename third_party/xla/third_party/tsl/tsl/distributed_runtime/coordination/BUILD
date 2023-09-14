load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")
load("//tsl:tsl.bzl", "if_oss", "set_external_visibility", "tsl_gpu_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_service_error_util",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/distributed_runtime:call_options",
        "//tsl/platform:status",
        "//tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_service",
    hdrs = ["coordination_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/protobuf:coordination_config_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# Keeping the implementation as a separate build target.
# This is an alwayslink library for statically registering "standalone" implementation.
# Other implementations of the service will be provided in the future.
tsl_gpu_library(
    name = "coordination_service_impl",
    srcs = ["coordination_service.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        "//tsl/distributed_runtime:call_options",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:random",
        "//tsl/platform:status",
        "//tsl/platform:strcat",
        "//tsl/platform:thread_annotations",
        "//tsl/protobuf:coordination_config_proto_cc",
        "//tsl/protobuf:coordination_service_proto_cc",
        "//tsl/util:device_name_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "test_device_proto",
    testonly = 1,
    srcs = ["test_device.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tsl_cc_test(
    name = "coordination_service_test",
    srcs = ["coordination_service_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        ":coordination_service_impl",
        ":test_device_proto_cc",
        "//tsl/distributed_runtime:call_options",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:random",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/protobuf:coordination_config_proto_cc",
        "//tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_gpu_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        ":coordination_service_error_util",
        "//tsl/distributed_runtime:call_options",
        "//tsl/framework:cancellation",
        "//tsl/lib/monitoring:gauge",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:random",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/platform:thread_annotations",
        "//tsl/protobuf:coordination_config_proto_cc",
        "//tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "coordination_service_agent_test",
    srcs = ["coordination_service_agent_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        ":coordination_service_agent",
        "//tsl/distributed_runtime:call_options",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:coordination_config_proto_cc_impl",
        "//tsl/protobuf:coordination_service_proto_cc_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "coordination_service_rpc_handler",
    srcs = ["coordination_service_rpc_handler.cc"],
    hdrs = [
        "coordination_service_rpc_handler.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_error_util",
        "//tsl/platform:casts",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "//tsl/platform:thread_annotations",
        "//tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "coordination_service_recoverable_job_test",
    srcs = ["coordination_service_recoverable_job_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_impl",
        "//tsl/distributed_runtime/rpc:async_service_interface",
        "//tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:coordination_config_proto_cc_impl",
        "//tsl/protobuf:coordination_service_proto_cc_impl",
        "//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
    visibility = ["//visibility:public"],
)
