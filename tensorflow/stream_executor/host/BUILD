# Description:
#   Host-platform specific StreamExecutor support code.

load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "host_platform_id",
    srcs = [
        "host_platform_id.cc",
    ],
    hdrs = [
        "host_platform_id.h",
    ],
    deps = [
        "//tensorflow/stream_executor:platform",
    ],
)

cc_library(
    name = "host_platform",
    srcs = [
        "host_platform.cc",
    ],
    hdrs = [
        "host_platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_gpu_executor",
        ":host_platform_id",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:executor_cache",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "host_stream",
    srcs = [
        "host_stream.cc",
    ],
    hdrs = [
        "host_stream.h",
    ],
    deps = [
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor:kernel",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "host_timer",
    srcs = [
        "host_timer.cc",
    ],
    hdrs = [
        "host_timer.h",
    ],
    deps = [
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/platform",
    ],
)

# TODO(22689637): Rename this target.
cc_library(
    name = "host_gpu_executor",
    srcs = [
        "host_gpu_executor.cc",
    ],
    hdrs = [
        "host_gpu_executor.h",
    ],
    deps = [
        ":host_platform_id",
        ":host_stream",
        ":host_timer",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:kernel",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_executor_pimpl",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "host_stream_test",
    srcs = ["host_stream_test.cc"],
    deps = [
        ":host_platform",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor:stream",
        "@com_google_absl//absl/synchronization",
    ],
)
