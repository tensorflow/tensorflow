load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: math_util_test.

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "math_util.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

exports_files([
    "math_util.h",
    "math_util_test.cc",
])
