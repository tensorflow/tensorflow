# Description:
# Graph C API.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "grappler_hdrs",
    hdrs = ["grappler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "grappler",
    srcs = ["grappler.cc"],
    hdrs = [
        "grappler.h",
        "grappler_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "grappler_test",
    srcs = ["grappler_test.cc"],
    deps = [
        ":grappler",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/costs:op_performance_data_cc",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
