# Description:
#   Host-platform specific StreamExecutor support code.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "host_platform_id",
    hdrs = [
        "host_platform_id.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor:platform",
    ],
)

cc_library(
    name = "host_platform",
    hdrs = [
        "host_platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_gpu_executor",
        ":host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/stream_executor:executor_cache",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "host_stream",
    hdrs = [
        "host_stream.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/host:host_stream",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor:kernel",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "host_timer",
    hdrs = [
        "host_timer.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/host:host_timer",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/platform",
    ],
)

# TODO(22689637): Rename this target.
cc_library(
    name = "host_gpu_executor",
    hdrs = [
        "host_gpu_executor.h",
    ],
    deps = [
        ":host_platform_id",
        ":host_stream",
        ":host_timer",
        "//tensorflow/compiler/xla/stream_executor/host:host_gpu_executor",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:profile_utils_cpu_utils",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:kernel",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_executor_pimpl",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)
