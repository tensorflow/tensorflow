load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/service/gpu:build_defs.bzl", "get_cub_sort_kernel_types")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_nccl", "internal_visibility", "nvtx_headers")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
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
    ] + if_cuda_is_configured(nvtx_headers()),
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
        ":custom_call_thunk",
        ":nccl_all_gather_thunk",
        ":nccl_all_reduce_thunk",
        ":nccl_api",
        ":nccl_clique_key",
        ":nccl_collective_broadcast_thunk",
        ":nccl_collective_thunk",
        "//xla:executable_run_options",
        "//xla:status",
        "//xla:types",
        "//xla:util",
        "//xla/ffi:call_frame",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_status_internal",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu/kernels:custom_kernel",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/tsl/concurrency:ref_count",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
# NCCL integration
#===-------------------------------------------------------------------------------------------===//

# A lot of build complexity below is because NCCL dependency might not always be available and we
# have `if_nccl` and `if_gpu_configured` that do not compose. NCCL header included directly in
# :nccl_api target and all other targets should use this header to launch collective operations.
# This allows to minimize the spreading of #ifdef all over the XLA code base.
alias(
    name = "nccl_api",
    actual = if_nccl(":_nccl_api_impl", ":_nccl_api_stub"),
)

cc_library(
    name = "_nccl_api_impl",
    srcs = if_gpu_is_configured(
        ["nccl_api.cc"],
        ["nccl_api_stub.cc"],
    ),
    hdrs = ["nccl_api.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nccl_clique_key",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
        "//xla/stream_executor/cuda:cuda_driver",
        "//xla/stream_executor/cuda:cuda_executor",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rccl",
        "//xla/stream_executor/rocm:rocm_driver",
        "//xla/stream_executor/rocm:rocm_executor",
    ]) + if_gpu_is_configured([
        "//xla/stream_executor/gpu:gpu_stream",
    ]),
)

cc_library(
    name = "_nccl_api_stub",
    srcs = ["nccl_api_stub.cc"],
    hdrs = ["nccl_api.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nccl_clique_key",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "nccl_clique",
    srcs = ["nccl_clique.cc"],
    hdrs = ["nccl_clique.h"],
    deps = [
        ":nccl_api",
        ":nccl_clique_key",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:status_macros",
        "//xla/service:global_device_id",
        "//xla/service:lockable",
        "//xla/service:rendezvous",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_clique_key",
    srcs = ["nccl_clique_key.cc"],
    hdrs = ["nccl_clique_key.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/service:global_device_id",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "nccl_clique_key_test",
    srcs = ["nccl_clique_key_test.cc"],
    deps = [
        ":nccl_clique_key",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/container:btree",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===-------------------------------------------------------------------------------------------===//
# XLA Thunks Runtime
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "address_computation_thunk",
    srcs = ["address_computation_thunk.cc"],
    hdrs = ["address_computation_thunk.h"],
    deps = [
        ":sequential_thunk",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:memory_allocation",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "address_computation_thunk_test",
    srcs = if_gpu_is_configured(["address_computation_thunk_test.cc"]),
    backend_tags = {
        "gpu_a100": if_google(["config-cuda-only"]),
        "gpu_v100": if_google(["config-cuda-only"]),
    },
    backends = [
        "gpu_a100",
        "gpu_v100",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":address_computation_thunk",
        ":custom_call_thunk",
        ":gemm_thunk",
        "//xla:shape_util",
        "//xla:types",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/algorithm:container",
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
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    deps = if_gpu_is_configured([
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:cusolver_context",
        "//xla/service/gpu:make_batch_pointers",
        "//xla/service/gpu/runtime:thunk",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ]),
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
        "//xla/service/gpu/runtime:thunk",
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
        "gpu_a100": if_google(["config-cuda-only"]),
        "gpu_v100": if_google(["config-cuda-only"]),
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu:variant_visitor",
        "//xla/service/gpu/runtime:thunk",
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
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_conv_runner",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/gpu/runtime:thunk",
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
        ":thunk",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/kernels:custom_kernel",
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "nccl_all_gather_thunk",
    srcs = ["nccl_all_gather_thunk.cc"],
    hdrs = ["nccl_all_gather_thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_collective_thunk",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/runtime:thunk",
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
        ":nccl_api",
        ":nccl_collective_thunk",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/runtime:thunk",
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
        ":nccl_api",
        ":nccl_collective_thunk",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu/runtime:thunk",
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
    name = "nccl_collective_broadcast_thunk",
    srcs = ["nccl_collective_broadcast_thunk.cc"],
    hdrs = ["nccl_collective_broadcast_thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_collective_thunk",
        "//xla:status",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_collective_permute_thunk",
    srcs = ["nccl_collective_permute_thunk.cc"],
    hdrs = ["nccl_collective_permute_thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_collective_thunk",
        ":nccl_p2p_thunk_common",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "//xla/tsl/concurrency:async_value",
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
    name = "nccl_collective_thunk",
    srcs = ["nccl_collective_thunk.cc"],
    hdrs = ["nccl_collective_thunk.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW=1",
    ]),
    deps = [
        ":nccl_api",
        ":nccl_clique",
        ":nccl_clique_key",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:global_device_id",
        "//xla/service:rendezvous",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu/runtime:thunk",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rccl",
    ]),
)

cc_library(
    name = "nccl_p2p_thunk_common",
    srcs = ["nccl_p2p_thunk_common.cc"],
    hdrs = ["nccl_p2p_thunk_common.h"],
    deps = [
        ":nccl_clique_key",
        ":nccl_collective_thunk",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_parser",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_recv_thunk",
    srcs = ["nccl_recv_thunk.cc"],
    hdrs = ["nccl_recv_thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_clique_key",
        ":nccl_collective_thunk",
        ":nccl_p2p_thunk_common",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:global_device_id",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nccl_send_thunk",
    srcs = ["nccl_send_thunk.cc"],
    hdrs = ["nccl_send_thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_clique_key",
        ":nccl_collective_thunk",
        ":nccl_p2p_thunk_common",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:global_device_id",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "thunk",
    srcs = ["thunk.cc"],
    hdrs = ["thunk.h"],
    deps = [
        ":nccl_api",
        ":nccl_clique",
        ":nccl_clique_key",
        "//xla:executable_run_options",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor",
        "//xla/translate/mhlo_to_hlo:location_exporter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/lib/gtl:int_type",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:make_batch_pointers",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ]),
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service/gpu/runtime:thunk",
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
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:kernel_arguments",
        "//xla/service/gpu/runtime:thunk",
        "//xla/stream_executor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)
