load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

#===-------------------------------------------------------------------------------------------===//
# Command Buffer Integration
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "command_buffer_cmd",
    srcs = ["command_buffer_cmd.cc"],
    hdrs = ["command_buffer_cmd.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "command_buffer_cmd_test",
    srcs = ["command_buffer_cmd_test.cc"],
    backends = ["gpu"],
    deps = [
        ":command_buffer_cmd",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_test_kernels",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===-------------------------------------------------------------------------------------------===//
# XLA Thunks Runtime
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:cusolver_context",
        "//xla/service/gpu:precompiled_kernels",
        "//xla/service/gpu:thunk",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["@local_tsl//tsl/platform:status"],
)

cc_library(
    name = "command_buffer_thunk",
    srcs = ["command_buffer_thunk.cc"],
    hdrs = ["command_buffer_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_buffer_cmd",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "command_buffer_thunk_test",
    srcs = ["command_buffer_thunk_test.cc"],
    backends = ["gpu"],
    deps = [
        ":command_buffer_cmd",
        ":command_buffer_thunk",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_test_kernels",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "custom_call_thunk",
    srcs = ["custom_call_thunk.cc"],
    hdrs = ["custom_call_thunk.h"],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service:custom_call_status_internal",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "fft_thunk",
    srcs = ["fft_thunk.cc"],
    hdrs = ["fft_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/strings:str_format",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:precompiled_kernels",
        "//xla/service/gpu:thunk",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["@local_tsl//tsl/platform:status"],
)
