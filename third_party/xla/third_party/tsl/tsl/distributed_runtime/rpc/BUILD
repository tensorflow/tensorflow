# Description:
#   RPC communication interfaces and implementations for TensorFlow.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("//tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "@local_xla//xla/tsl:internal",
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
        "//tsl/platform:mutex",
        "//tsl/platform:refcount",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = ["grpc_util.h"],
    deps = [
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:stringprintf",
        "//tsl/protobuf:distributed_runtime_payloads_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
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
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_channel_common",
    hdrs = ["grpc_channel_common.h"],
    deps = [
        ":grpc_util",
        "//tsl/platform:logging",
        "//tsl/platform:mutex",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "grpc_channel",
    srcs = ["grpc_channel.cc"],
    hdrs = ["grpc_channel.h"],
    deps = [
        ":grpc_channel_common",
        ":grpc_util",
        "//tsl/lib/gtl:map_util",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:numbers",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/protobuf:rpc_options_proto_cc",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:device_name_utils",
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
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:strcat",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:rpc_options_proto_cc_impl",
        "@local_xla//xla/tsl/util:device_name_utils",
    ],
)

cc_library(
    name = "grpc_state",
    hdrs = ["grpc_state.h"],
    deps = [
        ":grpc_client_cq_tag",
        ":grpc_util",
        "//tsl/distributed_runtime:call_options",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:strcat",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/util:env_var",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_client_cq_tag",
    srcs = [],
    hdrs = ["grpc_client_cq_tag.h"],
    deps = [
        "//tsl/platform:macros",
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
