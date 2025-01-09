# Description:
#   RPC communication interfaces and implementations for TensorFlow.

load("//xla/tsl:tsl.bzl", "internal_visibility")
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
    name = "async_service_interface",
    srcs = [],
    hdrs = ["async_service_interface.h"],
    deps = [],
)

cc_library(
    name = "grpc_call",
    srcs = [],
    hdrs = ["grpc_call.h"],
    deps = [
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:refcount",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = ["grpc_util.h"],
    deps = [
        "//xla/tsl/protobuf:distributed_runtime_payloads_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:stringprintf",
    ] + tsl_grpc_cc_dependencies(),
)

tsl_cc_test(
    name = "grpc_util_test",
    size = "small",
    srcs = ["grpc_util_test.cc"],
    tags = [
        "no_mac",
    ],
    deps = [
        ":grpc_util",
        ":test_request_proto_cc_impl",
        "//xla/tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_channel_common",
    hdrs = ["grpc_channel_common.h"],
    deps = [
        ":grpc_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "grpc_channel",
    srcs = ["grpc_channel.cc"],
    hdrs = ["grpc_channel.h"],
    deps = [
        ":grpc_channel_common",
        ":grpc_util",
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/protobuf:rpc_options_proto_cc",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
    ] + tsl_grpc_cc_dependencies(),
)

tsl_cc_test(
    name = "grpc_channel_test",
    size = "small",
    srcs = [
        "grpc_channel_test.cc",
    ],
    deps = [
        ":grpc_channel",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/protobuf:rpc_options_proto_cc_impl",
        "//xla/tsl/util:device_name_utils",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "grpc_state",
    hdrs = ["grpc_state.h"],
    deps = [
        ":grpc_client_cq_tag",
        ":grpc_util",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:strcat",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_client_cq_tag",
    srcs = [],
    hdrs = ["grpc_client_cq_tag.h"],
    deps = [
        "@local_tsl//tsl/platform:macros",
    ],
)

tf_proto_library(
    name = "test_request_proto",
    testonly = 1,
    srcs = ["test_request.proto"],
    create_java_proto = False,
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)
