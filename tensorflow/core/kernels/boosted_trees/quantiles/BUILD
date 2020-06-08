# Description:
#   This directory contains common quantile utilities used in boosted_trees.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

filegroup(
    name = "weighted_quantiles_hdrs",
    srcs = [
        "quantile_stream_resource.h",
        "weighted_quantiles_buffer.h",
        "weighted_quantiles_stream.h",
        "weighted_quantiles_summary.h",
    ],
)

# Quantiles

cc_library(
    name = "weighted_quantiles",
    srcs = [],
    hdrs = [
        "quantile_stream_resource.h",
        "weighted_quantiles_buffer.h",
        "weighted_quantiles_stream.h",
        "weighted_quantiles_summary.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "weighted_quantiles_buffer_test",
    size = "small",
    srcs = ["weighted_quantiles_buffer_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "weighted_quantiles_summary_test",
    size = "small",
    srcs = ["weighted_quantiles_summary_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "weighted_quantiles_stream_test",
    size = "small",
    srcs = ["weighted_quantiles_stream_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
