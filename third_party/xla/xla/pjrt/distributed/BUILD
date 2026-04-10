load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/pjrt:friends"]),
)

tf_proto_library(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        "//xla:types",
        "//xla:util",
        "//xla/pjrt/distributed/coordination:coordination_service",
        "//xla/pjrt/distributed/coordination:grpc_coordination_service_impl",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "topology_util_test",
    srcs = ["topology_util_test.cc"],
    deps = [
        ":in_memory_key_value_store",
        ":protocol_proto_cc",
        ":topology_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "client",
    srcs = [
        "client.cc",
    ],
    hdrs = [
        "client.h",
    ],
    deps = [
        ":key_value_store_interface",
        "//xla/pjrt/distributed/coordination:coordination_client",
        "//xla/pjrt/distributed/coordination:coordination_service",
        "//xla/pjrt/distributed/coordination:coordination_service_agent",
        "//xla/pjrt/distributed/coordination:grpc_coordination_client",
        "//xla/runtime:device_id",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "distributed",
    srcs = ["distributed.cc"],
    hdrs = ["distributed.h"],
    deps = [
        ":client",
        ":service",
        "//xla/tsl/platform:grpc_credentials",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "topology_util",
    srcs = ["topology_util.cc"],
    hdrs = ["topology_util.h"],
    deps = [
        ":key_value_store_interface",
        ":protocol_proto_cc",
        "//xla:util",
        "//xla/pjrt:utils",
        "//xla/service:gpu_topology_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "client_server_test",
    size = "medium",
    srcs = ["client_server_test.cc"],
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":client",
        ":distributed",
        ":protocol_proto_cc",
        ":service",
        ":topology_util",
        "//xla:status_macros",
        "//xla/runtime:device_id",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "key_value_store_interface",
    hdrs = ["key_value_store_interface.h"],
    deps = [
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "in_memory_key_value_store",
    srcs = ["in_memory_key_value_store.cc"],
    hdrs = ["in_memory_key_value_store.h"],
    deps = [
        ":key_value_store_interface",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
