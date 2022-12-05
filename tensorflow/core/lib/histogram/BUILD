load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/lib/monitoring:sampler uses histogram
        "//tensorflow/core/lib/monitoring:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    hdrs = ["histogram.h"],
    deps = [
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/histogram",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "histogram.h",
        "//tensorflow/tsl/lib/histogram:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
        "//tensorflow/tsl/lib/histogram:legacy_lib_histogram_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
