# Description: Utils for Tensorflow quantization.

load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")

# buildifier: disable=no-layering-check
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "uniform_quant_ops_params.cc",
        "uniform_quant_ops_params.h",
    ],
)

cc_library(
    name = "uniform_quant_ops_params",
    srcs = [
        "uniform_quant_ops_params.cc",
    ],
    hdrs = [
        "uniform_quant_ops_params.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "uniform_quant_ops_params_test",
    srcs = [
        "uniform_quant_ops_params_test.cc",
    ],
    deps = [
        ":uniform_quant_ops_params",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "uniform_quant_ops_attr_proto",
    srcs = ["uniform_quant_ops_attr.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)
