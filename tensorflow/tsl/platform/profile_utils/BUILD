# Description:
# profile_utils targets.

load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/tsl:tsl.bzl",
    "tsl_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/platform:__subpackages__",
        "//tensorflow/tsl:__pkg__",
        "//tensorflow/tsl/platform/default:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files(srcs = [
    "android_armv7a_cpu_utils_helper.cc",
    "android_armv7a_cpu_utils_helper.h",
    "clock_cycle_profiler.h",
    "cpu_utils.cc",
    "cpu_utils.h",
    "cpu_utils_test.cc",
    "i_cpu_utils_helper.h",
])

filegroup(
    name = "legacy_lib_internal_srcs",
    srcs = [
        "android_armv7a_cpu_utils_helper.cc",
        "clock_cycle_profiler.cc",
    ],
    visibility = [
        "//tensorflow/core/platform:__subpackages__",
        "//tensorflow/tsl/platform:__pkg__",
    ],
)

cc_library(
    name = "profile_utils_cpu_utils",
    srcs = [
        "android_armv7a_cpu_utils_helper.h",
        "cpu_utils.cc",
        "i_cpu_utils_helper.h",
    ],
    hdrs = [
        "cpu_utils.h",
    ],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)
