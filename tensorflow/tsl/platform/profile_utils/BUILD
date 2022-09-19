# Description:
# profile_utils targets.

load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",  # @unused
)

package(
    default_visibility = [
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/platform:__subpackages__",
        "//tensorflow/tsl:__pkg__",
        "//tensorflow/tsl/default:__pkg__",
        "//tensorflow/tsl/platform:__pkg__",
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
    copts = tf_copts(),
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/base",
    ],
    alwayslink = 1,
)
