load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "kernel_name_tracer_cuda",
    testonly = True,
    srcs = [
        "kernel_name_tracer_cuda.cc",
        "kernel_name_tracer_factory.h",
    ],
    hdrs = ["kernel_name_tracer.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/backends/profiler/gpu:cupti_collector",
        "//xla/backends/profiler/gpu:cupti_tracer",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/platform:platform_object_registry",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "kernel_name_tracer",
    testonly = True,
    srcs = [
        "kernel_name_tracer.cc",
        "kernel_name_tracer_factory.h",
    ],
    hdrs = ["kernel_name_tracer.h"],
    deps = [
        "//xla/stream_executor:platform",
        "//xla/stream_executor/platform:platform_object_registry",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
    ] + if_cuda_is_configured([
        # keep sorted
        ":kernel_name_tracer_cuda",
    ]),
)

xla_test(
    name = "kernel_name_tracer_test",
    srcs = ["kernel_name_tracer_test.cc"],
    # There is a leak in CUPTI which triggers the heap checker.
    args = if_google(["--heap_check="]),
    backends = [
        "gpu",
    ],
    tags = [
        "cuda-only",
    ],
    deps = [
        ":kernel_name_tracer",
        "//xla/backends/gpu/runtime:command",
        "//xla/backends/gpu/runtime:command_buffer_cmd",
        "//xla/backends/gpu/runtime:command_buffer_thunk",
        "//xla/backends/gpu/runtime:command_executor",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/stream_executor/gpu:gpu_test_kernels_fatbin",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
