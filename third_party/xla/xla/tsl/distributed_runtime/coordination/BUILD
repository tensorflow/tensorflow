load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "coordination_service_error_util",
    srcs = ["coordination_service_error_util.cc"],
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:regexp",
    ],
)

tsl_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    deps = [
        ":coordination_service_error_util",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    deps = [
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_service",
    srcs = ["coordination_service.cc"],
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_client",
        ":coordination_service_error_util",
        ":key_value_store",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:random",
    ],
)

tf_proto_library(
    name = "test_device_proto",
    testonly = 1,
    srcs = ["test_device.proto"],
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
        ":test_device_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:random",
    ],
)

cc_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    deps = [
        ":coordination_client",
        ":coordination_service_error_util",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/framework:cancellation",
        "//xla/tsl/lib/monitoring:gauge",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tsl_cc_test(
    name = "coordination_service_agent_test",
    srcs = ["coordination_service_agent_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service_agent",
        ":coordination_service_error_util",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "//xla/tsl/protobuf:coordination_service_proto_cc_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
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
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = ["key_value_store.h"],
    deps = [
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

tsl_cc_test(
    name = "key_value_store_test",
    srcs = ["key_value_store_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # no status matchers in OSS.
    deps = [
        ":key_value_store",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "coordination_service_recoverable_job_test",
    srcs = ["coordination_service_recoverable_job_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "//xla/tsl/protobuf:coordination_service_proto_cc_impl",
        "//xla/tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ] + tsl_grpc_cc_dependencies(),
)

tsl_cc_test(
    name = "client_server_test",
    size = "medium",
    srcs = ["client_server_test.cc"],
    shard_count = 4,
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "//xla/tsl/protobuf:coordination_service_proto_cc_impl",
        "//xla/tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ] + tsl_grpc_cc_dependencies(),
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
        "key_value_store.h",
    ],
)
