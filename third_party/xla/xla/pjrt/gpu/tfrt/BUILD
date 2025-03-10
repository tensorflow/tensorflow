load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/pjrt/gpu:package_groups.bzl", "xla_gpu_internal_packages")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# Integrate with PJRT rather than the GPU client directly.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

xla_gpu_internal_packages()

cc_library(
    name = "tfrt_gpu_client",
    srcs = ["tfrt_gpu_client.cc"],
    hdrs = ["tfrt_gpu_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    visibility = internal_visibility(["//xla/pjrt/gpu:legacy_gpu_client_users"]),
    deps = [
        ":stream_pool",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_stream_executor_device_description",
        "//xla/pjrt:utils",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:gpu_topology",
        "//xla/pjrt/gpu:gpu_topology_proto_cc",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:computation_placer_hdr",
        "//xla/service:executable",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "tfrt_gpu_client_test",
    srcs = ["tfrt_gpu_client_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":tfrt_gpu_client",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:gpu_plugin",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_event",
    srcs = ["gpu_event.cc"],
    hdrs = ["gpu_event.h"],
    deps = [
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gpu_event_test",
    srcs = ["gpu_event_test.cc"],
    deps = [
        ":gpu_event",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "stream_pool",
    srcs = ["stream_pool.cc"],
    hdrs = ["stream_pool.h"],
    deps = [
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "stream_pool_test",
    srcs = ["stream_pool_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":stream_pool",
        "//xla/service:backend",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
