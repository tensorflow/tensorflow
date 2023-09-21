load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("//xla:xla.bzl", "xla_cc_test")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "protocol",
    hdrs = ["protocol.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = ["key_value_store.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":key_value_store",
        ":protocol",
        ":protocol_cc_grpc_proto",
        ":topology_util",
        ":util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/distributed_runtime/coordination:coordination_service",
        "@local_tsl//tsl/distributed_runtime/coordination:coordination_service_impl",
        "@local_tsl//tsl/distributed_runtime/rpc:async_service_interface",
        "@local_tsl//tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

xla_cc_test(
    name = "topology_util_test",
    srcs = ["topology_util_test.cc"],
    deps = [
        ":protocol_proto_cc",
        ":topology_util",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
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
    visibility = ["//visibility:public"],
    deps = [
        ":protocol",
        ":protocol_cc_grpc_proto",
        ":util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/distributed_runtime/coordination:coordination_client",
        "@local_tsl//tsl/distributed_runtime/coordination:coordination_service_agent",
        "@local_tsl//tsl/distributed_runtime/coordination:coordination_service_error_util",
        "@local_tsl//tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/protobuf:coordination_config_proto_cc",
        "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "distributed",
    srcs = ["distributed.cc"],
    hdrs = ["distributed.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":service",
        "//xla:statusor",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "topology_util",
    srcs = ["topology_util.cc"],
    hdrs = ["topology_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":protocol_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "client_server_test",
    size = "small",
    srcs = ["client_server_test.cc"],
    deps = [
        ":client",
        ":distributed",
        ":protocol_proto_cc",
        ":service",
        "//xla:protobuf_util",
        "//xla:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + tsl_grpc_cc_dependencies(),
)
