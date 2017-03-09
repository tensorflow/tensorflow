# Description:
#   This directory contains a runtime O(nlog(log(n))) and
#   memory O(log^2(n)) efficient approximate quantiles
#   implementation allowing streaming and distributed
#   computation of quantiles on weighted data points.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "weighted_quantiles",
    srcs = [],
    hdrs = [
        "weighted_quantiles_buffer.h",
        "weighted_quantiles_stream.h",
        "weighted_quantiles_summary.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "weighted_quantiles_buffer_test",
    size = "small",
    srcs = ["weighted_quantiles_buffer_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "weighted_quantiles_summary_test",
    size = "small",
    srcs = ["weighted_quantiles_summary_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "weighted_quantiles_stream_test",
    size = "small",
    srcs = ["weighted_quantiles_stream_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
