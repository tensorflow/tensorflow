load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")
load("//tensorflow/tsl:tsl.bzl", "if_oss", "tsl_gpu_library")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    deps = [
        ":coordination_service_error_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    deps = [
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_service",
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_client",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc",
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
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_error_util",
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "//tensorflow/tsl/util:device_name_utils",
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
        ":coordination_service_impl",
        ":test_device_proto_cc",
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
)
