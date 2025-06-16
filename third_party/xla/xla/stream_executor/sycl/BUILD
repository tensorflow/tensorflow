# Description:
#   SYCL-platform specific StreamExecutor support code.

load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "sycl_platform_id",
    srcs = ["sycl_platform_id.cc"],
    hdrs = ["sycl_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "sycl_platform",
    srcs = ["sycl_platform.cc"],
    hdrs = ["sycl_platform.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

xla_cc_test(
    name = "sycl_platform_test",
    srcs = ["sycl_platform_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=If not building with SYCL, we don't have any tests linked.
    fail_if_no_test_selected = False,  # NOLINT=If not building with SYCL, we don't have any tests linked.
    deps = [
        ":sycl_platform",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = if_sycl_is_configured([":sycl_platform"]),
    alwayslink = 1,
)

cc_library(
    name = "sycl_rpath",
    linkopts = ["-Wl,-rpath,../local_config_sycl/sycl/sycl/lib"],
)

cc_library(
    name = "stream_executor_sycl",
    tags = [
        "gpu",
        "sycl-only",
    ],
    deps = [
        ":sycl_platform_id",
        ":sycl_rpath",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
    ] + if_static([":all_runtime"]),
)

cc_library(
    name = "sycl_solver_context",
    srcs = ["sycl_solver_context.cc"],
    hdrs = ["sycl_solver_context.h"],
    tags = [
        "gpu",
        "manual",
        "sycl-only",
    ],
    deps = [
        ":sycl_platform_id",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:gpu_solver_context",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/platform:platform_object_registry",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
