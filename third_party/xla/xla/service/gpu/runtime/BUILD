load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/service/gpu:build_defs.bzl", "get_cub_sort_kernel_types")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

#===-------------------------------------------------------------------------------------------===//
# Runtime tracing libraries
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "annotation",
    srcs = ["annotation.cc"],
    hdrs = ["annotation.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        "//xla:printer",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/lib:nvtx_utils",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

#===-------------------------------------------------------------------------------------------===//
# Command Buffer Integration
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "command_buffer_allocations",
    srcs = ["command_buffer_allocations.cc"],
    hdrs = ["command_buffer_allocations.h"],
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
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
    deps = [
        ":annotation",
        ":cudnn_thunk",
        ":custom_call_thunk",
        ":nccl_all_gather_thunk",
        ":nccl_all_reduce_thunk",
        "//xla:shape_util",
        "//xla:status",
        "//xla:types",
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_status_internal",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:global_device_id",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:nccl_api",
        "//xla/service/gpu:nccl_clique",
        "//xla/service/gpu:nccl_clique_key",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu:thunk",
        "//xla/service/gpu/kernels:custom_kernel",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "command_buffer_cmd_emitter",
    srcs = ["command_buffer_cmd_emitter.cc"],
    hdrs = ["command_buffer_cmd_emitter.h"],
    deps = [
        ":command_buffer_cmd",
        ":conditional_thunk",
        ":copy_thunk",
        ":cudnn_thunk",
        ":custom_call_thunk",
        ":gemm_thunk",
        ":kernel_thunk",
        ":memset_thunk",
        ":nccl_all_gather_thunk",
        ":nccl_all_reduce_thunk",
        ":replica_id_thunk",
        ":sequential_thunk",
        ":wait_for_streams_thunk",
        ":while_thunk",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
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
    ]) + [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "command_buffer_thunk",
    srcs = ["command_buffer_thunk.cc"],
    hdrs = ["command_buffer_thunk.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":annotation",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:profiler_lock",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
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
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "conditional_thunk",
    srcs = ["conditional_thunk.cc"],
    hdrs = ["conditional_thunk.h"],
    deps = [
        ":sequential_thunk",
        "//xla:status",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/service/gpu:variant_visitor",
        "//xla/stream_executor",
        "//xla/stream_executor:memory_allocation",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "convolution_thunk",
    srcs = ["convolution_thunk.cc"],
    hdrs = ["convolution_thunk.h"],
    deps = [
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_conv_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "copy_thunk",
    srcs = ["copy_thunk.cc"],
    hdrs = ["copy_thunk.h"],
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "cub_sort_thunk",
    srcs = if_gpu_is_configured(["cub_sort_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cub_sort_thunk.h"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor:device_memory",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@local_tsl//tsl/platform:errors",
    ] + ["//xla/service/gpu:cub_sort_kernel_" + suffix for suffix in get_cub_sort_kernel_types()]),
)

cc_library(
    name = "custom_call_thunk",
    srcs = ["custom_call_thunk.cc"],
    hdrs = ["custom_call_thunk.h"],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "fft_thunk",
    srcs = ["fft_thunk.cc"],
    hdrs = ["fft_thunk.h"],
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "fused_mha_thunk",
    srcs = ["fused_mha_thunk.cc"],
    hdrs = ["fused_mha_thunk.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_fused_mha_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gemm_thunk",
    srcs = ["gemm_thunk.cc"],
    hdrs = ["gemm_thunk.h"],
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "gpublas_lt_matmul_thunk",
    srcs = if_gpu_is_configured(["gpublas_lt_matmul_thunk.cc"]),
    hdrs = if_gpu_is_configured(["gpublas_lt_matmul_thunk.h"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:thunk",
        "//xla:status",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "@local_tsl//tsl/platform:logging",
    ]),
)

cc_library(
    name = "infeed_thunk",
    srcs = ["infeed_thunk.cc"],
    hdrs = ["infeed_thunk.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:io_feed_manager",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "kernel_thunk",
    srcs = ["kernel_thunk.cc"],
    hdrs = ["kernel_thunk.h"],
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
        "@com_google_absl//absl/status",
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
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "nccl_all_gather_thunk",
    srcs = ["nccl_all_gather_thunk.cc"],
    hdrs = ["nccl_all_gather_thunk.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:nccl_api",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_all_reduce_thunk",
    srcs = ["nccl_all_reduce_thunk.cc"],
    hdrs = ["nccl_all_reduce_thunk.h"],
    deps = [
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:nccl_api",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_all_to_all_thunk",
    srcs = ["nccl_all_to_all_thunk.cc"],
    hdrs = ["nccl_all_to_all_thunk.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:nccl_api",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_collective_permute_thunk",
    srcs = ["nccl_collective_permute_thunk.cc"],
    hdrs = ["nccl_collective_permute_thunk.h"],
    deps = [
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:nccl_api",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "norm_thunk",
    srcs = ["norm_thunk.cc"],
    hdrs = ["norm_thunk.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_norm_runner",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "outfeed_thunk",
    srcs = ["outfeed_thunk.cc"],
    hdrs = ["outfeed_thunk.h"],
    deps = [
        "//xla:util",
        "//xla/service/gpu:io_feed_manager",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "replica_id_thunk",
    srcs = ["replica_id_thunk.cc"],
    hdrs = ["replica_id_thunk.h"],
    deps = [
        "//xla/service:buffer_assignment",
        "//xla/service:global_device_id",
        "//xla/service/gpu:thunk",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "sequential_thunk",
    srcs = ["sequential_thunk.cc"],
    hdrs = ["sequential_thunk.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":annotation",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "send_recv_thunk",
    srcs = ["send_recv_thunk.cc"],
    hdrs = ["send_recv_thunk.h"],
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
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
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
    ]) + [
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "while_thunk",
    srcs = ["while_thunk.cc"],
    hdrs = ["while_thunk.h"],
    deps = [
        ":sequential_thunk",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "wait_for_streams_thunk",
    srcs = ["wait_for_streams_thunk.cc"],
    hdrs = ["wait_for_streams_thunk.h"],
    deps = [
        "//xla/service:global_device_id",
        "//xla/service/gpu:thunk",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "cudnn_thunk",
    srcs = ["cudnn_thunk.cc"],
    hdrs = ["cudnn_thunk.h"],
    deps = [
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:kernel_arguments",
        "//xla/service/gpu:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)
