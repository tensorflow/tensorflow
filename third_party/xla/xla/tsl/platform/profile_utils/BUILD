# Description:
# profile_utils targets.

load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/platform:__subpackages__",
        "@local_tsl//tsl:__pkg__",
        "//xla/tsl/platform/default:__pkg__",
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
        "@local_tsl//tsl/platform:__pkg__",
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
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ],
    alwayslink = 1,
)
