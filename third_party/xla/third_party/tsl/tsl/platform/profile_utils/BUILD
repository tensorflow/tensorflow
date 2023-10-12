# Description:
# profile_utils targets.

load("//tsl:tsl.bzl", "set_external_visibility")
load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tsl:tsl.bzl",
    "tsl_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "android_armv7a_cpu_utils_helper.cc",
        "android_armv7a_cpu_utils_helper.h",
        "clock_cycle_profiler.h",
        "cpu_utils.cc",
        "cpu_utils.h",
        "cpu_utils_test.cc",
        "i_cpu_utils_helper.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_internal_srcs",
    srcs = [
        "android_armv7a_cpu_utils_helper.cc",
        "clock_cycle_profiler.cc",
    ],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)
