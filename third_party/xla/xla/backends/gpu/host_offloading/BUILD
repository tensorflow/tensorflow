load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "gpu_host_offloading_allocator",
    srcs = ["gpu_host_offloading_allocator.cc"],
    hdrs = ["gpu_host_offloading_allocator.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:util",
        "//xla/backends/cpu:alignment",
        "//xla/core/host_offloading:host_offloading_allocator",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:bfc_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:numbers",
    ],
)

xla_test(
    name = "gpu_host_offloading_allocator_test",
    srcs = ["gpu_host_offloading_allocator_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_host_offloading_allocator",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
