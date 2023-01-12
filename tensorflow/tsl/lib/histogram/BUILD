load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/lib/histogram:__pkg__",
        "//tensorflow/tsl/lib/monitoring:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    srcs = ["histogram.cc"],
    hdrs = ["histogram.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "histogram.cc",
        "histogram.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/histogram:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/histogram:__pkg__",
    ],
)

tsl_cc_test(
    name = "histogram_test",
    srcs = [
        "histogram_test.cc",
    ],
    deps = [
        ":histogram",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
    ],
)
