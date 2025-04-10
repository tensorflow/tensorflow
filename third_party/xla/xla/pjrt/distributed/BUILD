load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
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
        ":topology_util",
        ":util",
        "//xla:types",
        "//xla:util",
        "//xla/tsl/distributed_runtime/coordination:coordination_service",
        "//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:random",
    ] + tsl_grpc_cc_dependencies(),
)

xla_cc_test(
    name = "topology_util_test",
    srcs = ["topology_util_test.cc"],
    deps = [
        ":in_memory_key_value_store",
        ":protocol_proto_cc",
        ":topology_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
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
        ":util",
        "//xla/tsl/distributed_runtime/coordination:coordination_client",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//xla/tsl/protobuf:coordination_config_proto_cc",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@com_google_absl//absl/status",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "distributed",
    srcs = ["distributed.cc"],
    hdrs = ["distributed.h"],
    deps = [
        ":client",
        ":service",
        "//xla/tsl/platform:grpc_credentials",
        "@com_google_absl//absl/status:statusor",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "topology_util",
    srcs = ["topology_util.cc"],
    hdrs = ["topology_util.h"],
    deps = [
        ":key_value_store_interface",
        ":protocol_proto_cc",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:utils",
        "//xla/pjrt/gpu:gpu_topology_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
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
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "key_value_store_interface",
    hdrs = ["key_value_store_interface.h"],
    deps = [
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
