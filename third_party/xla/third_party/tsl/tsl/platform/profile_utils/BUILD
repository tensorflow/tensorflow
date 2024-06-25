# Description:
# profile_utils targets.

load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "@local_xla//xla/stream_executor:__subpackages__",
        "//tensorflow/core/platform:__subpackages__",
        "//tsl:__pkg__",
        "//tsl/platform/default:__pkg__",
    ]),
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
    visibility = internal_visibility([
        "//tensorflow/core/platform:__subpackages__",
        "//tsl/platform:__pkg__",
    ]),
)

cc_library(
    name = "profile_utils_cpu_utils",
    srcs = [
        "android_armv7a_cpu_utils_helper.h",
        "cpu_utils.cc",
    ],
    hdrs = [
        "cpu_utils.h",
        "i_cpu_utils_helper.h",
    ],
    copts = tsl_copts(),
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)
