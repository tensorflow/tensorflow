# Description:
#   SYCL-platform specific StreamExecutor support code.

load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
    "sycl_library",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_oss", "internal_visibility", "tsl_copts")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
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
    deps = ["//xla/stream_executor:platform_id"],
)

cc_library(
    name = "sycl_platform",
    srcs = ["sycl_platform.cc"],
    hdrs = ["sycl_platform.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_executor",
        ":sycl_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

xla_test(
    name = "sycl_platform_test",
    srcs = ["sycl_platform_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_platform",
        ":sycl_platform_id",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
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
    linkopts = if_oss(["-Wl,-rpath,../local_config_sycl/sycl/sycl/lib"]),
)

cc_library(
    name = "stream_executor_sycl",
    tags = [
        "gpu",
        "oneapi-only",
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
    name = "sycl_status",
    srcs = ["sycl_status.cc"],
    hdrs = ["sycl_status.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_proto_library(
    name = "oneapi_compute_capability_proto",
    srcs = ["oneapi_compute_capability.proto"],
    make_default_target_header_only = True,
)

cc_library(
    name = "oneapi_compute_capability",
    srcs = ["oneapi_compute_capability.cc"],
    hdrs = ["oneapi_compute_capability.h"],
    deps = [
        ":oneapi_compute_capability_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "oneapi_compute_capability_test",
    srcs = ["oneapi_compute_capability_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":oneapi_compute_capability",
        ":oneapi_compute_capability_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "sycl_status_test",
    srcs = ["sycl_status_test.cc"],
    deps = [
        ":sycl_status",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

sycl_library(
    name = "sycl_stream",
    srcs = ["sycl_stream.cc"],
    hdrs = ["sycl_stream.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_context",
        ":sycl_event",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "sycl_stream_test",
    srcs = ["sycl_stream_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":sycl_event",
        ":sycl_executor",
        ":sycl_platform_id",
        ":sycl_stream",
        "//xla/backends/gpu/runtime:kernel_thunk",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sycl_solver_context",
    srcs = ["sycl_solver_context.cc"],
    hdrs = ["sycl_solver_context.h"],
    tags = ["gpu"],
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

sycl_library(
    name = "sycl_kernel",
    srcs = ["sycl_kernel.cc"],
    hdrs = ["sycl_kernel.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_metadata",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "sycl_kernel_test",
    srcs = ["sycl_kernel_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_platform_id",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sycl_executor",
    srcs = ["sycl_executor.cc"],
    hdrs = ["sycl_executor.h"],
    compatible_with = [],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_context",
        ":sycl_event",
        ":sycl_gpu_runtime",
        ":sycl_kernel",
        ":sycl_stream",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_test(
    name = "sycl_executor_test",
    srcs = ["sycl_executor_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
        "pjrt_migration_candidate",
    ],
    use_legacy_runtime = True,
    deps = [
        ":sycl_executor",
        "//xla/backends/gpu/runtime:kernel_thunk",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:intel_gpu_compiler",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

sycl_library(
    name = "sycl_event",
    srcs = ["sycl_event.cc"],
    hdrs = ["sycl_event.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "sycl_event_test",
    srcs = ["sycl_event_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_event",
        ":sycl_platform_id",
        "//xla/stream_executor:platform_manager",
        "@com_google_googletest//:gtest_main",
    ],
)

sycl_library(
    name = "sycl_timer",
    srcs = ["sycl_timer.cc"],
    hdrs = ["sycl_timer.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_event",
        ":sycl_gpu_runtime",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

xla_test(
    name = "sycl_timer_test",
    srcs = ["sycl_timer_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_executor",
        ":sycl_timer",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:typed_kernel_factory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

sycl_library(
    name = "sycl_gpu_runtime",
    srcs = ["sycl_gpu_runtime.cc"],
    hdrs = ["sycl_gpu_runtime.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_test(
    name = "sycl_gpu_runtime_test",
    srcs = ["sycl_gpu_runtime_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_gpu_runtime",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

sycl_library(
    name = "sycl_context",
    srcs = ["sycl_context.cc"],
    hdrs = ["sycl_context.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_gpu_runtime",
        "//xla/stream_executor/gpu:context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "sycl_context_test",
    srcs = ["sycl_context_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_context",
        ":sycl_platform_id",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_googletest//:gtest_main",
    ],
)
