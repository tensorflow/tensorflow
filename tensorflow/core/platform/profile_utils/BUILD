# Description:
# profile_utils targets.

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",  # @unused
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/default:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files(srcs = [
    "android_armv7a_cpu_utils_helper.h",
    "clock_cycle_profiler.h",
    "cpu_utils.h",
    "i_cpu_utils_helper.h",
])

filegroup(
    name = "legacy_lib_internal_srcs",
    srcs = [
        "//tensorflow/tsl/platform/profile_utils:legacy_lib_internal_srcs",
    ],
    visibility = ["//tensorflow/core/platform:__pkg__"],
)

cc_library(
    name = "profile_utils_cpu_utils",
    srcs = [
        "android_armv7a_cpu_utils_helper.h",
        "i_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    hdrs = [
        "cpu_utils.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)
