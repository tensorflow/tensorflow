load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")

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

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
    ],
)
