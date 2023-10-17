# Description:
#   Host-platform specific StreamExecutor support code.

load("//xla:xla.bzl", "xla_cc_test")
load("@local_tsl//tsl:tsl.bzl", "set_external_visibility")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor:platform",
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
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:denormal",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:setround",
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
    visibility = ["//visibility:public"],
    deps = [
        ":host_platform_id",
        ":host_stream",
        "//xla/stream_executor",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform/profile_utils:profile_utils_cpu_utils",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "host_stream_test",
    srcs = ["host_stream_test.cc"],
    deps = [
        ":host_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
