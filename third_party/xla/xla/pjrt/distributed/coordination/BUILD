load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "coordination_service_proto",
    srcs = ["coordination_service.proto"],
    has_services = 1,
    create_grpc_library = True,
    create_java_proto = False,
    create_kotlin_proto = False,
    create_service = True,
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_library(
    name = "coordination_service_error_util",
    srcs = ["coordination_service_error_util.cc"],
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        ":coordination_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:regexp",
    ],
)

xla_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    deps = [
        ":coordination_service_error_util",
        ":coordination_service_proto_cc",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    deps = [
        ":coordination_service_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "coordination_service",
    srcs = ["coordination_service.cc"],
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_service_error_util",
        ":coordination_service_proto_cc",
        ":key_value_store",
        "//xla/service:global_device_id",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:random",
    ],
)

tf_proto_library(
    name = "test_device_proto",
    testonly = 1,
    srcs = ["test_device.proto"],
)

xla_cc_test(
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
        ":coordination_service_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
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
        "@tsl//tsl/platform:random",
    ],
)

cc_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        ":coordination_service_proto_cc",
        "//xla:util",
        "//xla/runtime:device_id",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/framework:cancellation",
        "//xla/tsl/lib/monitoring:gauge",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:random",
    ],
)

xla_cc_test(
    name = "coordination_service_agent_test",
    srcs = ["coordination_service_agent_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service_agent",
        ":coordination_service_error_util",
        ":coordination_service_proto_cc_impl",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
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
        ":coordination_service_proto_cc",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_lite",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = ["key_value_store.h"],
    deps = [
        ":coordination_service_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "key_value_store_test",
    srcs = ["key_value_store_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # no status matchers in OSS.
    deps = [
        ":coordination_service_proto_cc",
        ":key_value_store",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "client_server_test",
    size = "medium",
    srcs = ["client_server_test.cc"],
    shard_count = 4,
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_proto_cc_impl",
        ":grpc_coordination_client",
        ":grpc_coordination_service_impl",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:coordination_config_proto_cc_impl",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
        "key_value_store.h",
    ],
)

cc_library(
    name = "grpc_coordination_client",
    srcs = ["grpc_coordination_client.cc"],
    hdrs = ["grpc_coordination_client.h"],
    deps = [
        ":coordination_client",
        ":coordination_service_proto_cc",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/rpc:grpc_channel",
        "//xla/tsl/distributed_runtime/rpc:grpc_client_cq_tag",
        "//xla/tsl/distributed_runtime/rpc:grpc_state",
        "//xla/tsl/distributed_runtime/rpc:grpc_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "grpc_coordination_service_impl",
    srcs = ["grpc_coordination_service_impl.cc"],
    hdrs = ["grpc_coordination_service_impl.h"],
    deps = [
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_cc_grpc_proto",
        ":coordination_service_proto_cc",
        ":coordination_service_rpc_handler",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc:grpc_call",
        "//xla/tsl/distributed_runtime/rpc:grpc_util",
        "//xla/tsl/platform:env",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)
