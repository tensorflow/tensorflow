load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: math_util_test.

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "@local_xla//xla/tsl/lib/math:math_util",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "math_util.h",
    ],
    visibility = ["//visibility:public"],
)

exports_files([
    "math_util.h",
])
