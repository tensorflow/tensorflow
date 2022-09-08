# Description:
#   Host-platform specific StreamExecutor support code.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
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
        "//tensorflow/compiler/xla/stream_executor:platform",
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
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
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
        "//tensorflow/compiler/xla/stream_executor:kernel",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:lib_internal",
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
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/platform",
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
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:kernel",
        "//tensorflow/compiler/xla/stream_executor:rng",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core/platform:profile_utils_cpu_utils",
        "//tensorflow/tsl/platform:platform_port",
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
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/synchronization",
    ],
)
