load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

# StreamExecutor clients that can depend on targets defined in stream_executor package.
package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

#===--------------------------------------------------------------------------------------------===#
# Exporting headers for Tensorflow
#===--------------------------------------------------------------------------------------------===#

# Tensorflow device memory allocators are aliases for StreamExecutor allocators, we export
# headers for Tensorflow to build shared libraries in OSS.

filegroup(
    name = "device_mem_allocator_headers",
    srcs = [
        "device_mem_allocator.h",
    ],
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor integrations
#===--------------------------------------------------------------------------------------------===#

# Libraries integrating StreamExecutor into framework APIs, e.g. plugging StreamExecutor device
# memory allocator into Tensorflow.

cc_library(
    name = "tf_allocator_adapter",
    srcs = ["tf_allocator_adapter.cc"],
    hdrs = ["tf_allocator_adapter.h"],
    deps = [
        "//xla:shape_util",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "device_mem_allocator",
    hdrs = [
        "device_mem_allocator.h",
    ],
    deps = [
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:device_id",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "stream_executor_allocator",
    srcs = ["stream_executor_allocator.cc"],
    hdrs = [
        "stream_executor_allocator.h",
    ],
    deps = [
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "stream_executor_allocator_test",
    srcs = ["stream_executor_allocator_test.cc"],
    deps = [
        ":stream_executor_allocator",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ] + if_google([
        "//xla/tsl/framework:allocator",
    ]),
)

xla_cc_test(
    name = "tf_allocator_adapter_test",
    srcs = ["tf_allocator_adapter_test.cc"],
    deps = [
        ":tf_allocator_adapter",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ] + if_google([
        "//xla/tsl/framework:allocator",
    ]),
)
