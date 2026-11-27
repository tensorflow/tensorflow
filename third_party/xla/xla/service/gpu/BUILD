# Description:
#   GPU-specific components in XLA service implementation.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tests:build_defs.bzl",
    "AMD_GPU_ANY_BACKEND",
    "GPU_DEFAULT_BACKENDS",
    "NVIDIA_GPU_DEFAULT_BACKENDS",
    "xla_test",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

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

#Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

tf_proto_library(
    name = "backend_configs",
    srcs = ["backend_configs.proto"],
    make_default_target_header_only = True,
    deps = [
        "//xla:autotuning_proto",
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/codegen/xtile:xtile_config_proto",
        "//xla/stream_executor/gpu:blas_lt_matmul_algorithm_proto",
        "//xla/tsl/protobuf:dnn_proto",
    ],
)

xla_cc_test(
    name = "backend_configs_test",
    srcs = ["backend_configs_test.cc"],
    deps = [
        ":backend_configs_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_executable_run_options",
    hdrs = ["gpu_executable_run_options.h"],
    compatible_with = get_compatible_with_portable(),
    tags = if_google(["avoid_dep"]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/backends/gpu/runtime:gpu_executable_run_options",
    ],
)

cc_library(
    name = "gpu_memory_space_assignment",
    srcs = ["gpu_memory_space_assignment.cc"],
    hdrs = ["gpu_memory_space_assignment.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_ops_utils",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:collective_ops_utils",
        "//xla/service:decision",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_memory_space_assignment_test",
    srcs = ["gpu_memory_space_assignment_test.cc"],
    deps = [
        ":gpu_memory_space_assignment",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "launch_dimensions",
    srcs = [
        "launch_dimensions.cc",
    ],
    hdrs = [
        "launch_dimensions.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":launch_dimensions_proto_cc",
        "//xla:shape_util",
        "//xla:util",
        "//xla/runtime:work_cluster",
        "//xla/runtime:work_dimensions",
        "//xla/runtime:work_group",
        "//xla/runtime:work_item",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "launch_dimensions_proto",
    srcs = ["launch_dimensions.proto"],
    deps = ["//xla/stream_executor:launch_dim_proto"],
)

xla_cc_test(
    name = "launch_dimensions_test",
    srcs = ["launch_dimensions_test.cc"],
    deps = [
        ":launch_dimensions",
        ":launch_dimensions_proto_cc",
        "//xla/runtime:work_cluster",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    backends = GPU_DEFAULT_BACKENDS,
    tags = [
        # TODO(intel-tf): Remove it when macro substitutions for SYCL are available in xla/stream_executor/sycl/*.
        "no-oneapi",
    ],
    deps = [
        "//xla:array",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi:execution_context",
        "//xla/ffi:ffi_api",
        "//xla/ffi:invoke",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:client_library_test_runner_mixin",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_insertion_test",
    srcs = ["gpu_copy_insertion_test.cc"],
    deps = [
        ":gpu_device_info_for_tests",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms:copy_insertion",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "target_util",
    srcs = ["target_util.cc"],
    hdrs = ["target_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TargetParser",
    ],
)

xla_cc_test(
    name = "target_util_test",
    srcs = ["target_util_test.cc"],
    deps = [
        ":target_util",
        "//xla:xla_data_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "gpu_device_info_for_tests",
    # This is *not* a test library because it is used in a cc_binary which is used for testing but
    # test_only libraries are not allowed in cc_binaries.
    testonly = 0,
    srcs = ["gpu_device_info_for_tests.cc"],
    hdrs = ["gpu_device_info_for_tests.h"],
    compatible_with = get_compatible_with_portable(),
    local_defines = if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/gpu:core_info",
        "//xla/stream_executor/rocm:rocm_compute_capability",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_cuda",
    srcs = ["custom_kernel_emitter_cuda.cc"],
    hdrs = ["custom_kernel_emitter.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":ir_emitter_context",
        ":kernel_call",
        ":launch_dimensions",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_rocm",
    srcs = ["custom_kernel_emitter_rocm.cc"],
    hdrs = ["custom_kernel_emitter.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":ir_emitter_context",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_sycl",
    srcs = ["custom_kernel_emitter_sycl_stub.cc"],
    hdrs = ["custom_kernel_emitter.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":ir_emitter_context",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter",
    hdrs = ["custom_kernel_emitter.h"],
    tags = ["gpu"],
    deps = if_cuda_is_configured([
        ":ptx_custom_kernel_emitter_cuda",
    ]) + if_rocm_is_configured([
        ":ptx_custom_kernel_emitter_rocm",
    ]) + if_sycl_is_configured([
        ":ptx_custom_kernel_emitter_sycl",
    ]) + ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "ir_emitter_context",
    hdrs = ["ir_emitter_context.h"],
    deps = [
        ":execution_stream_assignment",
        ":kernel_reuse_cache",
        ":mlir_context_pool",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/backends/gpu/runtime:host_execute_thunk",
        "//xla/backends/gpu/runtime:thunk_id",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:call_inliner",
        "//xla/service:buffer_assignment",
        "//xla/service:gpu_topology",
        "//xla/service:name_uniquer",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "thunk_emitter",
    srcs = ["thunk_emitter.cc"],
    hdrs = ["thunk_emitter.h"],
    tags = ["gpu"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":dense_data_intermediate",
        ":execution_stream_assignment",
        ":gpu_conv_runner",
        ":gpu_hlo_ordering",
        ":gpu_norm_runner",
        ":hlo_fusion_analysis",
        ":ir_emission_utils",
        ":ir_emitter_context",
        ":kernel_reuse_cache",
        ":launch_dimensions",
        ":matmul_utils",
        ":ptx_custom_kernel_emitter",
        ":stream_executor_util",
        ":triton_call",
        "//xla:comparison_util",
        "//xla:future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/codegen:pad_to_static",
        "//xla/backends/gpu/codegen:rng_get_and_update_state",
        "//xla/backends/gpu/codegen:slice_to_dynamic",
        "//xla/backends/gpu/codegen:sort",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/codegen/triton:triton_kernel_source",
        "//xla/backends/gpu/codegen/triton:triton_wrapper_result",
        "//xla/backends/gpu/codegen/triton:xtile_compiler",
        "//xla/backends/gpu/ffi:ffi_attributes_from_backend_config",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_copy",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_fusion",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_emitter_context",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_handler_registry",
        "//xla/backends/gpu/runtime:all_gather_thunk",
        "//xla/backends/gpu/runtime:all_reduce_thunk",
        "//xla/backends/gpu/runtime:all_to_all_thunk",
        "//xla/backends/gpu/runtime:async_execution",
        "//xla/backends/gpu/runtime:async_thunk",
        "//xla/backends/gpu/runtime:collective_broadcast_thunk",
        "//xla/backends/gpu/runtime:collective_group_thunk",
        "//xla/backends/gpu/runtime:collective_permute_thunk",
        "//xla/backends/gpu/runtime:collective_reduce_thunk",
        "//xla/backends/gpu/runtime:collective_thunk",
        "//xla/backends/gpu/runtime:conditional_thunk",
        "//xla/backends/gpu/runtime:convolution_reorder_thunk",
        "//xla/backends/gpu/runtime:convolution_thunk",
        "//xla/backends/gpu/runtime:cudnn_thunk",
        "//xla/backends/gpu/runtime:custom_call_thunk",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:device_to_host_copy_thunk",
        "//xla/backends/gpu/runtime:dynamic_slice_fusion_v2_thunk",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/backends/gpu/runtime:fft_thunk",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/backends/gpu/runtime:gpublas_lt_matmul_thunk",
        "//xla/backends/gpu/runtime:host_execute_thunk",
        "//xla/backends/gpu/runtime:host_send_recv_thunk",
        "//xla/backends/gpu/runtime:host_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:infeed_thunk",
        "//xla/backends/gpu/runtime:legacy_custom_call_thunk",
        "//xla/backends/gpu/runtime:memset_thunk",
        "//xla/backends/gpu/runtime:norm_thunk",
        "//xla/backends/gpu/runtime:outfeed_thunk",
        "//xla/backends/gpu/runtime:ragged_all_to_all_thunk",
        "//xla/backends/gpu/runtime:recv_thunk",
        "//xla/backends/gpu/runtime:replica_id_thunk",
        "//xla/backends/gpu/runtime:rng_seed_thunk",
        "//xla/backends/gpu/runtime:select_k_exec_ffi",
        "//xla/backends/gpu/runtime:send_thunk",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:topk",
        "//xla/backends/gpu/runtime:triangular_solve_thunk",
        "//xla/backends/gpu/runtime:while_thunk",
        "//xla/backends/gpu/target_config:cudnn_device_props_json",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_kernel_source",
        "//xla/codegen/emitters:computation_fingerprint",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/codegen/xtile:block_level_parameters",
        "//xla/core/host_offloading:host_offloading_executable_proto_cc",
        "//xla/ffi:attribute_map",
        "//xla/ffi:attributes",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:transforms_gpu_passes",
        "//xla/runtime:device_id",
        "//xla/service:buffer_assignment",
        "//xla/service:call_graph",
        "//xla/service:collective_opt_utils",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_proto_cc",
        "//xla/service:shaped_slice",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor/gpu:blas_lt_matmul_algorithm",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/concurrency:future",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
        "@tsl//tsl/platform:human_readable_json",
    ],
)

cc_library(
    name = "kernel_call",
    srcs = ["kernel_call.cc"],
    hdrs = ["kernel_call.h"],
    deps = [
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "kernel_call_test",
    srcs = ["kernel_call_test.cc"],
    deps = [
        ":kernel_call",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "triton_call",
    srcs = ["triton_call.cc"],
    hdrs = ["triton_call.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "triton_call_test",
    srcs = ["triton_call_test.cc"],
    deps = [
        ":triton_call",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

bool_flag(
    name = "enable_xlir",
    build_setting_default = if_google(True, False),
)

cc_library(
    name = "gpu_executable",
    hdrs = ["gpu_executable.h"],
    tags = if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu/runtime:gpu_executable",
    ],
)

tf_proto_library(
    name = "gpu_executable_proto",
    srcs = ["gpu_executable.proto"],
    deps = [
        ":dense_data_intermediate_proto",
        "//xla:shape_util_proto",
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/backends/gpu/runtime:thunk_proto",
        "//xla/service:gpu_topology_proto",
        "//xla/service:hlo_proto",
        "//xla/service/cpu:executable_proto",
        "//xla/stream_executor:device_description_proto",
        "//xla/stream_executor/abi:executable_abi_version_proto",
    ],
)

cc_library(
    name = "dense_data_intermediate",
    srcs = ["dense_data_intermediate.cc"],
    hdrs = ["dense_data_intermediate.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":dense_data_intermediate_proto_cc",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "dense_data_intermediate_proto",
    srcs = ["dense_data_intermediate.proto"],
)

xla_cc_test(
    name = "dense_data_intermediate_test",
    srcs = ["dense_data_intermediate_test.cc"],
    deps = [
        ":dense_data_intermediate",
        ":dense_data_intermediate_proto_cc",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/codegen:ir_emission_utils",
        "//xla/codegen/xtile:xtile_config_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:matmul_indexing_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:regexp",
    ],
)

xla_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reduction_utils",
    srcs = ["reduction_utils.cc"],
    hdrs = ["reduction_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "reduction_utils_test",
    srcs = ["reduction_utils_test.cc"],
    deps = [
        ":reduction_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cublas_cudnn",
    srcs = ["cublas_cudnn.cc"],
    hdrs = ["cublas_cudnn.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dynamic_slicing_utils",
    srcs = ["dynamic_slicing_utils.cc"],
    hdrs = ["dynamic_slicing_utils.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:call_graph",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "matmul_utils",
    srcs = ["matmul_utils.cc"],
    hdrs = ["matmul_utils.h"],
    deps = [
        ":backend_configs_cc",
        "//xla:autotuning_proto_cc",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/hlo/transforms:dot_algorithm_rewriter",
        "//xla/hlo/ir:hlo",
        "//xla/service:algorithm_util",
        "//xla/service:matmul_indexing_utils",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:engine_options",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "matmul_utils_test",
    srcs = ["matmul_utils_test.cc"],
    deps = [
        ":matmul_utils",
        "//xla:autotuning_proto_cc",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_conv_runner",
    srcs = ["gpu_conv_runner.cc"],
    hdrs = ["gpu_conv_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":gpu_conv_runner_proto_cc",
        ":stream_executor_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:lazy_op_runner",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

tf_proto_library(
    name = "gpu_conv_runner_proto",
    srcs = ["gpu_conv_runner.proto"],
    deps = [
        ":backend_configs",
        "//xla:xla_data_proto",
        "//xla/tsl/protobuf:dnn_proto",
    ],
)

xla_cc_test(
    name = "gpu_conv_runner_test",
    srcs = ["gpu_conv_runner_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_conv_runner",
        ":gpu_conv_runner_proto_cc",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_norm_runner",
    srcs = ["gpu_norm_runner.cc"],
    hdrs = ["gpu_norm_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":gpu_norm_runner_proto_cc",
        ":stream_executor_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:lazy_op_runner",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "gpu_norm_runner_test",
    srcs = ["gpu_norm_runner_test.cc"],
    deps = [
        ":gpu_norm_runner",
        ":gpu_norm_runner_proto_cc",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "gpu_norm_runner_proto",
    srcs = ["gpu_norm_runner.proto"],
    deps = [
        ":backend_configs",
        "//xla:xla_data_proto",
    ],
)

cc_library(
    name = "cudnn_support_utils",
    srcs = ["cudnn_support_utils.cc"],
    hdrs = ["cudnn_support_utils.h"],
    deps = [
        ":cublas_cudnn",
        "//xla:shape_util",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "cudnn_support_utils_test",
    srcs = ["cudnn_support_utils_test.cc"],
    deps = [
        ":cudnn_support_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

tf_proto_library(
    name = "kernel_reuse_cache_proto",
    srcs = ["kernel_reuse_cache.proto"],
    deps = [":launch_dimensions_proto"],
)

cc_library(
    name = "target_constants",
    hdrs = ["target_constants.h"],
)

cc_library(
    name = "gpu_transfer_manager",
    srcs = ["gpu_transfer_manager.cc"],
    hdrs = ["gpu_transfer_manager.h"],
    deps = [
        ":io_feed_manager",
        ":target_constants",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:compiler",
        "//xla/service:generic_transfer_manager",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/stream_executor/sycl:sycl_platform_id",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Core",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:statusor",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_float_support",
    srcs = ["gpu_float_support.cc"],
    hdrs = ["gpu_float_support.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/triton:support",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:float_support",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "compile_module_to_llvm_ir",
    srcs = [
        "compile_module_to_llvm_ir.cc",
    ],
    hdrs = [
        "compile_module_to_llvm_ir.h",
    ],
    tags = ["gpu"],
    deps = [
        ":execution_stream_assignment",
        ":gpu_hlo_ordering",
        ":gpu_memory_space_assignment",
        ":ir_emitter_context",
        ":kernel_reuse_cache_proto_cc",
        ":metrics",
        ":mlir_context_pool",
        ":thunk_emitter",
        "//xla:future",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/backends/gpu/runtime:runtime_intrinsics",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen:llvm_kernel_source",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:dump",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_proto_cc",
        "//xla/service:logical_buffer",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "compile_module_to_llvm_ir_test",
    srcs = ["compile_module_to_llvm_ir_test.cc"],
    tags = ["gpu"],
    deps = [
        ":compile_module_to_llvm_ir",
        ":mlir_context_pool",
        ":target_constants",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/codegen:cubin_custom_kernel_compiler",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:buffer_value",
        "//xla/service:gpu_topology",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional_ref",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "pre_scheduling_copy_insertion_pipeline",
    srcs = ["pre_scheduling_copy_insertion_pipeline.cc"],
    hdrs = ["pre_scheduling_copy_insertion_pipeline.h"],
    deps = [
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/backends/gpu/hlo/transforms:alias_passthrough_params",
        "//xla/backends/gpu/hlo/transforms:constant_fill_copy_rewriter",
        "//xla/backends/gpu/hlo/transforms:copy_fusion",
        "//xla/backends/gpu/hlo/transforms:sanitize_constant_names",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:copy_insertion",
        "//xla/hlo/transforms:layout_assignment",
        "//xla/hlo/transforms:loop_schedule_linearizer",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_verifier",
        "//xla/stream_executor:device_description",
    ],
)

cc_library(
    name = "gpu_pipeline",
    srcs = ["gpu_pipeline.cc"],
    hdrs = ["gpu_pipeline.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/pass:hlo_pass_context",
        "//xla/hlo/pass:hlo_pass_options",
        "//xla/hlo/pass:hlo_pipeline_editor",
        "//xla/hlo/pass:hlo_pipeline_proto_cc",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

# Links the HloPassRegistry registrations of the passes in
# GetDefaultCompilationPipeline. gpu_pipeline_test checks that the list is
# complete.
cc_library(
    name = "gpu_pipeline_passes",
    deps = [
        "//xla/backends/gpu/hlo/transforms:algebraic_simplifier",
        "//xla/backends/gpu/hlo/transforms:composite_rewriter",
        "//xla/backends/gpu/hlo/transforms:cudnn_custom_call_converter",
        "//xla/backends/gpu/hlo/transforms:erase_xla_late_frontend_attributes",
        "//xla/core/host_offloading:hlo_host_device_type_call_wrapper",
        "//xla/hlo/transforms:call_inliner",
        "//xla/hlo/transforms:convert_memory_placement_to_internal_annotations",
        "//xla/hlo/transforms:nullary_function_wrap_inliner",
        "//xla/hlo/transforms:sharding_propagation",
        "//xla/hlo/transforms:sharding_remover",
        "//xla/hlo/transforms:topk_rewriter",
        "//xla/hlo/transforms/collectives:collective_permute_cse",
        "//xla/hlo/transforms/expanders:gather_expander",
        "//xla/hlo/transforms/expanders:scatter_expander",
        "//xla/hlo/transforms/simplifiers:conditional_canonicalizer",
        "//xla/hlo/transforms/simplifiers:conditional_simplifier",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_constant_folding",
        "//xla/hlo/transforms/simplifiers:hlo_constant_splitter",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:recognize_reduce_window",
        "//xla/hlo/transforms/simplifiers:reshape_mover",
        "//xla/hlo/transforms/simplifiers:sort_simplifier",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/transforms/simplifiers:while_loop_constant_sinking",
        "//xla/hlo/transforms/simplifiers:while_loop_simplifier",
        "//xla/hlo/transforms/simplifiers:zero_sized_hlo_elimination",
        "//xla/service/spmd:collective_permute_motion",
        "//xla/service/spmd:spmd_partitioner",
        "//xla/service/spmd/shardy:shardy_manual_computation_inliner",
        "//xla/service/spmd/shardy:shardy_xla_pass",
    ],
)

xla_cc_test(
    name = "gpu_pipeline_test",
    srcs = ["gpu_pipeline_test.cc"],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_pipeline",
        ":gpu_pipeline_passes",  # keep -- buildcleaner: keep: links HloPassRegistry factories
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/core/host_offloading:hlo_host_device_type_call_wrapper",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_context",
        "//xla/hlo/pass:hlo_pass_options",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/pass:hlo_pass_registry",
        "//xla/hlo/pass:hlo_pipeline_builder",
        "//xla/hlo/pass:hlo_pipeline_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service/spmd/shardy:constants",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_compiler",
    hdrs = ["gpu_compiler.h"],
    tags = ["gpu"] + if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu:gpu_compiler",
    ],
)

xla_test(
    name = "gpu_offloading_test",
    srcs = ["gpu_offloading_test.cc"],
    backends = GPU_DEFAULT_BACKENDS,
    deps = [
        ":backend_configs_cc",
        "//xla:autotune_results_proto_cc",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/hlo/transforms:stream_attribute_annotator",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/transforms/simplifiers:hlo_rematerialization",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:buffer_value",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "nvptx_compiler",
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ] + if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu:nvptx_compiler",
    ],
)

cc_library(
    name = "nvptx_compiler_impl",
    hdrs = ["nvptx_compiler.h"],
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ] + if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu:nvptx_compiler_impl",
    ],
)

xla_test(
    name = "ptx_compilation_test",
    srcs = [
        "ptx_compilation_test.cc",
    ],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
        "gb200": ["broken"],  # TODO(b/505574653): re-enable when fixed.
        "gb300": ["broken"],  # TODO(b/505574653): re-enable when fixed.
    },
    backends = NVIDIA_GPU_DEFAULT_BACKENDS,
    tags = [
        "cuda-only",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/gpu:nvptx_compiler_impl",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:nvjitlink_support",
        "//xla/stream_executor/cuda:ptx_compilation_method",
        "//xla/stream_executor/cuda:ptx_compiler_support",
        "//xla/stream_executor/cuda:ptx_linking_method",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "gpu_aot_compilation_test",
    srcs = [
        "gpu_aot_compilation_test.cc",
    ],
    backends = GPU_DEFAULT_BACKENDS,
    tags = [
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
    ],
    deps = if_cuda_is_configured([
        "//xla/backends/gpu:nvptx_compiler_impl",
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/backends/gpu:amdgpu_compiler_impl",
    ]) + [
        ":gpu_transfer_manager",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:support",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/service:compiled_module",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:literal_test_util",
        "//xla/util/split_proto:split_executable_and_options_writer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)

xla_test(
    name = "gb200_cross_compilation_test",
    srcs = [
        "gb200_cross_compilation_test.cc",
    ],
    backends = ["gb200"],
    deps = [
        ":gpu_executable_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/autotuner:in_memory_store",
        "//xla/backends/gpu/autotuner:autotuner_cache",
        "//xla/backends/gpu/target_config",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiled_module",
        "//xla/service:compiler",
        "//xla/service:gpu_topology",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "amdgpu_compiler",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu:amdgpu_compiler",
    ],
)

cc_library(
    name = "amdgpu_compiler_impl",
    hdrs = ["amdgpu_compiler.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google(["avoid_dep"]),
    deps = [
        "//xla/backends/gpu:amdgpu_compiler_impl",
    ],
)

cc_library(
    name = "xfeed_queue",
    hdrs = ["xfeed_queue.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "io_feed_manager",
    srcs = [
        "infeed_manager.cc",
        "outfeed_manager.cc",
    ],
    hdrs = [
        "infeed_manager.h",
        "outfeed_manager.h",
    ],
    deps = [
        ":xfeed_queue",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla/stream_executor:device_address_handle",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:notification",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_transformer_test",
    srcs = ["while_transformer_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "stream_executor_util",
    srcs = ["stream_executor_util.cc"],
    hdrs = ["stream_executor_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tsl_copts(),
    deps = [
        ":cublas_cudnn",
        ":launch_dimensions",
        "//xla:autotuning_proto_cc",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:per_device_state",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:data_type",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_metadata",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/stream_executor/gpu:repeat_buffer_kernel",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "//xla/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ],
)

xla_cc_test(
    name = "stream_executor_util_test",
    srcs = ["stream_executor_util_test.cc"],
    deps = [
        ":cublas_cudnn",
        ":stream_executor_util",
        "//xla:autotuning_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:mock_stream_executor",
        "//xla/tsl/platform:env",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "//xla/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_asm_opts_util",
    srcs = ["gpu_asm_opts_util.cc"],
    hdrs = ["gpu_asm_opts_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:xla_proto_cc",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ],
)

xla_cc_test(
    name = "gpu_asm_opts_util_test",
    srcs = ["gpu_asm_opts_util_test.cc"],
    deps = [
        ":gpu_asm_opts_util",
        "//xla:xla_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_fusion_analysis",
    srcs = ["hlo_fusion_analysis.cc"],
    hdrs = ["hlo_fusion_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        ":reduction_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen:hlo_fusion_spec",
        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:decision",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "hlo_fusion_analysis_test",
    srcs = ["hlo_fusion_analysis_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_device_info_for_tests",
        ":hlo_fusion_analysis",
        ":ir_emission_utils",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_fusible",
    srcs = ["gpu_fusible.cc"],
    hdrs = ["gpu_fusible.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        ":hlo_fusion_analysis",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":reduction_utils",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/hlo/transforms:instruction_fusion",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "gpu_fusible_test",
    srcs = ["gpu_fusible_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        ":hlo_fusion_analysis",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms:instruction_fusion",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "float_support_test",
    srcs = ["float_support_test.cc"],
    backends = [
        "a100",
        "h100",
        "b200",
        AMD_GPU_ANY_BACKEND,
    ],
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "conv_layout_normalization_test",
    srcs = ["conv_layout_normalization_test.cc"],
    backends = GPU_DEFAULT_BACKENDS,
    deps = [
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "gpu_float_support_test",
    srcs = ["gpu_float_support_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_float_support",
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/simplifiers:float_normalization",
        "//xla/service:hlo_verifier",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//xla/tsl/lib/monitoring:counter",
        "//xla/tsl/lib/monitoring:gauge",
        "//xla/tsl/lib/monitoring:sampler",
    ],
)

cc_library(
    name = "conv_utils",
    srcs = ["conv_utils.cc"],
    hdrs = ["conv_utils.h"],
    deps = [
        ":backend_configs_cc",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "conv_utils_test",
    srcs = ["conv_utils_test.cc"],
    deps = [
        ":conv_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_proto_cc",
        "//xla/service:shape_inference",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "conv_layout_normalization",
    srcs = ["conv_layout_normalization.cc"],
    hdrs = ["conv_layout_normalization.h"],
    deps = [
        ":cublas_cudnn",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "dot_algorithm_support_test",
    srcs = ["dot_algorithm_support_test.cc"],
    backends = [
        "v100",
        "a100",
        "h100",
        "b200",
        AMD_GPU_ANY_BACKEND,
    ],
    tags = ["no_mac"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "kernel_reuse_cache",
    srcs = ["kernel_reuse_cache.cc"],
    hdrs = ["kernel_reuse_cache.h"],
    deps = [
        ":kernel_reuse_cache_proto_cc",
        ":launch_dimensions",
        ":launch_dimensions_proto_cc",
        "//xla:future",
        "//xla:status_macros",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/codegen/emitters:computation_fingerprint",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/gpu:tma_metadata",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:sorted_range",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "kernel_reuse_cache_test",
    srcs = ["kernel_reuse_cache_test.cc"],
    deps = [
        ":kernel_reuse_cache",
        ":kernel_reuse_cache_proto_cc",
        ":launch_dimensions",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/stream_executor:launch_dim",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "determinism_test",
    srcs = ["determinism_test.cc"],
    backends = GPU_DEFAULT_BACKENDS,
    deps = [
        "//xla:literal",
        "//xla:xla_proto_cc",
        "//xla/backends/core/autotuner:backends_proto_cc",
        "//xla/backends/gpu/autotuner:autotuner_cache",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:mock_stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_symbol_repository",
    hdrs = ["gpu_symbol_repository.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:autotune_results_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:symbol_repository",
    ],
)

cc_library(
    name = "execution_stream_assignment",
    srcs = ["execution_stream_assignment.cc"],
    hdrs = ["execution_stream_assignment.h"],
    deps = [
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_copy",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/hlo/transforms/hlo_scheduling:gpu_latency_hiding_scheduler",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:collective_opt_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "execution_stream_assignment_test",
    srcs = ["execution_stream_assignment_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":execution_stream_assignment",
        "//xla:side_effect_util",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "ptx_compile_options_from_debug_options",
    srcs = ["ptx_compile_options_from_debug_options.cc"],
    hdrs = ["ptx_compile_options_from_debug_options.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/stream_executor/cuda:compilation_options",
    ],
)

xla_cc_test(
    name = "ptx_compile_options_from_debug_options_test",
    srcs = ["ptx_compile_options_from_debug_options_test.cc"],
    deps = [
        ":ptx_compile_options_from_debug_options",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/stream_executor/cuda:compilation_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flag_utils",
    hdrs = ["flag_utils.h"],
    deps = [
        "//xla/backends/gpu/hlo/transforms:double_buffer_loop_unrolling",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/collectives:collective_pipeliner",
        "//xla/hlo/transforms/hlo_scheduling:latency_hiding_scheduler",
        "//xla/service:hlo_module_config",
    ],
)

xla_cc_test(
    name = "flag_utils_test",
    srcs = ["flag_utils_test.cc"],
    deps = [
        ":flag_utils",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/transforms:double_buffer_loop_unrolling",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/collectives:collective_pipeliner",
        "//xla/hlo/transforms/hlo_scheduling:latency_hiding_scheduler",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_module_config",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_hlo_ordering",
    srcs = ["gpu_hlo_ordering.cc"],
    hdrs = ["gpu_hlo_ordering.h"],
    deps = [
        "//xla:shape_util",
        "//xla/ffi:ffi_registry",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_hlo_ordering_test",
    size = "small",
    srcs = ["gpu_hlo_ordering_test.cc"],
    deps = [
        ":gpu_hlo_ordering",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",  # buildcleaner: keep
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mlir_context_pool",
    srcs = ["mlir_context_pool.cc"],
    hdrs = ["mlir_context_pool.h"],
    deps = [
        "//xla/runtime:object_pool",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "mlir_context_pool_test",
    srcs = ["mlir_context_pool_test.cc"],
    deps = [
        ":mlir_context_pool",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)
