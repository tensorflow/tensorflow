load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
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
    name = "command_buffer_allocations",
    srcs = ["command_buffer_allocations.cc"],
    hdrs = ["command_buffer_allocations.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "command_buffer_cmd",
    srcs = ["command_buffer_cmd.cc"],
    hdrs = ["command_buffer_cmd.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_buffer_allocations",
        "//xla:status",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:nccl_utils",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu:thunk",
        "//xla/service/gpu/kernels:custom_kernel",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "command_buffer_cmd_emitter",
    srcs = ["command_buffer_cmd_emitter.cc"],
    hdrs = ["command_buffer_cmd_emitter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_buffer_cmd",
        ":copy_thunk",
        ":kernel_thunk",
        ":memset_thunk",
        ":sequential_thunk",
        ":while_thunk",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gemm_thunk",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "command_buffer_cmd_test",
    srcs = if_gpu_is_configured(["command_buffer_cmd_test.cc"]),
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":command_buffer_cmd",
        "//xla:status",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
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
        "//xla/service/gpu:make_batch_pointers",
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
        ":command_buffer_allocations",
        ":command_buffer_cmd",
        "//xla:status",
        "//xla:statusor",
        "//xla/service:buffer_assignment",  # build_cleaner: keep
        "//xla/service/gpu:buffer_allocations",  # build_cleaner: keep
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

xla_test(
    name = "command_buffer_thunk_test",
    srcs = if_gpu_is_configured(["command_buffer_thunk_test.cc"]),
    backend_tags = {
        "gpu_a100": ["config-cuda-only"],
        "gpu_v100": ["config-cuda-only"],
    },
    backends = [
        "gpu_a100",
        "gpu_v100",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":command_buffer_allocations",
        ":command_buffer_cmd",
        ":command_buffer_thunk",
        "//xla:shape_util",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "conditional_thunk",
    srcs = ["conditional_thunk.cc"],
    hdrs = ["conditional_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sequential_thunk",
        "//xla:status",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "convolution_thunk",
    srcs = ["convolution_thunk.cc"],
    hdrs = ["convolution_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_conv_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "copy_thunk",
    srcs = ["copy_thunk.cc"],
    hdrs = ["copy_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@llvm-project//mlir:IR",
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
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla/ffi:call_frame",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_status_internal",
        "//xla/service:executable",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
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
    name = "for_thunk",
    srcs = ["for_thunk.cc"],
    hdrs = ["for_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sequential_thunk",
        "//xla:status",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "fused_mha_thunk",
    srcs = ["fused_mha_thunk.cc"],
    hdrs = ["fused_mha_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_fused_mha_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "infeed_thunk",
    srcs = ["infeed_thunk.cc"],
    hdrs = ["infeed_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:io_feed_manager",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
    ],
)

cc_library(
    name = "kernel_thunk",
    srcs = ["kernel_thunk.cc"],
    hdrs = ["kernel_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:kernel_arguments",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu:thunk",
        "//xla/service/gpu/kernels:custom_kernel",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "memset_thunk",
    srcs = ["memset_thunk.cc"],
    hdrs = ["memset_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
    ],
)

cc_library(
    name = "norm_thunk",
    srcs = ["norm_thunk.cc"],
    hdrs = ["norm_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_norm_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "sequential_thunk",
    srcs = ["sequential_thunk.cc"],
    hdrs = ["sequential_thunk.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "send_recv_thunk",
    srcs = ["send_recv_thunk.cc"],
    hdrs = ["send_recv_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service:global_device_id",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
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
        "//xla/service/gpu:make_batch_pointers",
        "//xla/service/gpu:thunk",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["@local_tsl//tsl/platform:status"],
)

cc_library(
    name = "while_thunk",
    srcs = ["while_thunk.cc"],
    hdrs = ["while_thunk.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sequential_thunk",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@local_tsl//tsl/platform:errors",
    ],
)
