# Description:
#   GPU-specific components in XLA service implementation.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//tensorflow/tsl:tsl.bzl", "if_google", "if_nccl", "tsl_copts", "tsl_gpu_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_hipblaslt",
    "if_rocm_is_configured",
)
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
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
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto",
        "//tensorflow/compiler/xla:autotuning_proto",
    ],
)

xla_cc_test(
    name = "backend_configs_test",
    srcs = ["backend_configs_test.cc"],
    deps = [
        ":backend_configs_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "gpu_executable_run_options",
    srcs = ["gpu_executable_run_options.cc"],
    hdrs = ["gpu_executable_run_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "gpu_constants",
    hdrs = ["gpu_constants.h"],
)

cc_library(
    name = "gpu_types",
    hdrs = ["gpu_types.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "@com_google_absl//absl/types:variant",
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
        ":gpu_device_info",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "custom_call_thunk",
    srcs = ["custom_call_thunk.cc"],
    hdrs = ["custom_call_thunk.h"],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
    deps = [
        ":buffer_allocations",
        ":thunk",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "custom_call_test",
    srcs = if_gpu_is_configured(["custom_call_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/runtime:module",
        "//tensorflow/compiler/xla/runtime:module_registry",
        "//tensorflow/compiler/xla/runtime/ffi:ffi_api",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_cc_test(
    name = "gpu_copy_insertion_test",
    srcs = if_gpu_is_configured(["gpu_copy_insertion_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = if_gpu_is_configured([
        ":gpu_compiler",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ]),
)

cc_library(
    name = "hlo_to_ir_bindings",
    srcs = ["hlo_to_ir_bindings.cc"],
    hdrs = ["hlo_to_ir_bindings.h"],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "target_util",
    srcs = ["target_util.cc"],
    hdrs = ["target_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TargetParser",
    ],
)

xla_cc_test(
    name = "target_util_test",
    srcs = ["target_util_test.cc"],
    deps = [
        ":target_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:test",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "gpu_device_info",
    srcs = ["gpu_device_info.cc"],
    hdrs = ["gpu_device_info.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_description_proto_cc",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "gpu_device_info_for_tests",
    testonly = 1,
    srcs = ["gpu_device_info_for_tests.cc"],
    hdrs = ["gpu_device_info_for_tests.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":gpu_device_info",
    ],
)

xla_cc_test(
    name = "gpu_device_info_test",
    srcs = ["gpu_device_info_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_device_info",
        ":gpu_device_info_for_tests",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_gpu_executor_header",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_gpu_executor_header",
    ]),
)

cc_library(
    name = "ir_emitter_context",
    srcs = ["ir_emitter_context.cc"],
    hdrs = ["ir_emitter_context.h"],
    deps = [
        ":gpu_constants",
        ":gpu_device_info",
        ":gpu_executable",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ir_emitter_unnested",
    srcs = ["ir_emitter_unnested.cc"],
    hdrs = ["ir_emitter_unnested.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]) + if_rocm_hipblaslt([
        "TF_HIPBLASLT=1",
    ]),
    deps = [
        ":backend_configs_cc",
        ":custom_call_thunk",
        ":fft_thunk",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_device_info",
        ":gpu_executable",
        ":gpu_fused_mha_runner",
        ":gpu_fusible",
        ":hlo_fusion_analysis",
        ":hlo_to_ir_bindings",
        ":ir_emission_utils",
        ":ir_emitter",
        ":ir_emitter_context",
        ":kernel_arguments",
        ":kernel_reuse_cache",
        ":launch_dimensions",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":parallel_loop_emitter",
        ":reduction_utils",
        ":target_util",
        ":thunk",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/mlir_hlo:transforms_gpu_passes",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service/gpu/fusions",
        "//tensorflow/compiler/xla/service/gpu/fusions:thunk_util",
        "//tensorflow/compiler/xla/service/gpu/fusions:tiling_util",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:dynamic_update_slice_util",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:sort_util",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:attribute_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:human_readable_json",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ] + if_gpu_is_configured([
        ":triangular_solve_thunk",
        ":cholesky_thunk",
        ":cublas_lt_matmul_thunk",
        ":ir_emitter_triton",
    ]),
)

cc_library(
    name = "ir_emitter",
    srcs = [
        "elemental_ir_emitter.cc",
        "ir_emitter.cc",
        "ir_emitter_nested.cc",
    ],
    hdrs = [
        "elemental_ir_emitter.h",
        "ir_emitter.h",
        "ir_emitter_nested.h",
        "kernel_mapping_scheme.h",
    ],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":hlo_fusion_analysis",
        ":hlo_to_ir_bindings",
        ":ir_emitter_context",
        ":kernel_reuse_cache",
        ":target_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_builder_mixin",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:math_ops",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ir_emitter_triton",
    srcs = if_cuda_is_configured(["ir_emitter_triton.cc"]) + if_rocm_hipblaslt([
        "ir_emitter_triton.cc",
    ]),
    hdrs = if_gpu_is_configured(["ir_emitter_triton.h"]),
    deps = [
        ":gemm_rewriter_triton",
        ":gpu_device_info",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":matmul_utils",
        ":target_util",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonDialects",
        "@triton//:TritonTransforms",
    ] + if_cuda_is_configured([
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonLLVMIR",
        "@triton//:TritonToTritonGPU",
    ]),
)

xla_test(
    name = "ir_emitter_triton_test",
    srcs = if_cuda_is_configured(["ir_emitter_triton_test.cc"]),
    backend_tags = {"gpu": [
        "requires-gpu-sm70",
    ]},
    backends = [
        "gpu",
    ],
    shard_count = 10,
    tags = ["nomac"],
    deps = [
        ":backend_configs_cc",
        ":gpu_device_info",
        ":gpu_device_info_for_tests",
        ":ir_emission_utils",
        ":ir_emitter_triton",
        ":launch_dimensions",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
    ],
)

xla_test(
    name = "ir_emitter_triton_large_test",
    srcs = if_cuda_is_configured(["ir_emitter_triton_large_test.cc"]),
    backend_tags = {"gpu": [
        "requires-gpu-sm70",
    ]},
    backends = [
        "gpu",
    ],
    tags = [
        "large",
        "no_oss",
        "nomac",
        "notap",
        "requires-mem:16g",
    ],
    deps = [
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

xla_test(
    name = "ir_emitter_triton_parametrized_test",
    srcs = if_cuda_is_configured(["ir_emitter_triton_parametrized_test.cc"]),
    backend_tags = {"gpu": [
        "requires-gpu-sm70",
    ]},
    backends = [
        "gpu",
    ],
    shard_count = 10,
    tags = ["nomac"],
    deps = [
        ":gemm_rewriter_triton",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "triton_autotuner",
    srcs = if_cuda_is_configured(["triton_autotuner.cc"]),
    hdrs = if_cuda_is_configured(["triton_autotuner.h"]),
    deps = if_cuda_is_configured([
        ":autotuner_compile_util",
        ":autotuner_util",
        ":backend_configs_cc",
        ":buffer_comparator",
        ":gemm_rewriter_triton",
        ":gemm_rewriter",
        ":gpu_device_info",
        ":gpu_float_support",
        ":gpu_fusible",
        ":instruction_fusion",
        ":ir_emission_utils",
        ":stream_executor_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:float_normalization",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/tsl/platform:blocking_counter",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util/proto:proto_utils",
    ]),
)

xla_test(
    name = "triton_autotuner_test",
    srcs = if_cuda_is_configured(["triton_autotuner_test.cc"]),
    backend_tags = {"gpu": [
        "requires-gpu-sm70",
    ]},
    backends = [
        "gpu",
    ],
    tags = [
        "nomac",
    ],
    deps = [
        ":autotuner_util",
        ":backend_configs_cc",
        ":gemm_rewriter_triton",
        ":triton_autotuner",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:platform_port",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parallel_loop_emitter",
    srcs = ["parallel_loop_emitter.cc"],
    hdrs = ["parallel_loop_emitter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":launch_dimensions",
        ":target_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "buffer_allocations",
    srcs = ["buffer_allocations.cc"],
    hdrs = ["buffer_allocations.h"],
    deps = [
        ":gpu_constants",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "thunk",
    srcs = ["thunk.cc"],
    hdrs = ["thunk.h"],
    deps = [
        ":buffer_allocations",
        ":gpu_executable_run_options",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:IR",
    ],
)

tsl_gpu_library(
    name = "nccl_collective_thunks",
    srcs = [
        "nccl_all_gather_thunk.cc",
        "nccl_all_reduce_thunk.cc",
        "nccl_all_to_all_thunk.cc",
        "nccl_collective_permute_thunk.cc",
        "nccl_collective_thunk.cc",
        "nccl_p2p_thunk_common.cc",
        "nccl_recv_thunk.cc",
        "nccl_send_thunk.cc",
    ],
    hdrs = [
        "nccl_all_gather_thunk.h",
        "nccl_all_reduce_thunk.h",
        "nccl_all_to_all_thunk.h",
        "nccl_collective_permute_thunk.h",
        "nccl_collective_thunk.h",
        "nccl_p2p_thunk_common.h",
        "nccl_recv_thunk.h",
        "nccl_send_thunk.h",
    ],
    # Override tsl_gpu_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        ":nccl_utils",
        ":thunk",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:attribute_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
    ],
)

# Empty library to implement nested dependency conditions.
cc_library(name = "empty")

# If NCCL/RCCL is supported, this target '#defines XLA_ENABLE_XCCL' and
# provides a header which #includes NCCL/RCCL.
alias(
    name = "nccl_utils",
    actual = if_nccl(":_nccl_utils", ":empty"),
)

# Do not depend on this target, but rather depend on :nccl_utils.
tsl_gpu_library(
    name = "_nccl_utils",
    srcs = if_gpu_is_configured(["nccl_utils.cc"]),
    hdrs = if_gpu_is_configured(["nccl_utils.h"]),
    # Override tsl_gpu_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    defines = if_gpu_is_configured(["XLA_ENABLE_XCCL"]),
    tags = ["manual"],  # Only builds with if_nccl().
    deps = if_gpu_is_configured([
        ":gpu_executable_run_options",
        ":thunk",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service:rendezvous",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:env",
    ]) + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rccl",
    ]),
)

# TODO(b/244780257): Remove this config.
bool_flag(
    name = "enable_xlir",
    build_setting_default = if_google(True, False),
)

cc_library(
    name = "non_atomically_upgradeable_rw_lock",
    srcs = [],
    hdrs = [
        "non_atomically_upgradeable_rw_lock.h",
    ],
    deps = [
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "non_atomically_upgradeable_rw_lock_test",
    srcs = ["non_atomically_upgradeable_rw_lock_test.cc"],
    deps = [
        ":non_atomically_upgradeable_rw_lock",
        "//tensorflow/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_executable",
    srcs = [
        "conditional_thunk.cc",
        "convolution_thunk.cc",
        "copy_thunk.cc",
        "for_thunk.cc",
        "fused_mha_thunk.cc",
        "gpu_executable.cc",
        "infeed_thunk.cc",
        "kernel_thunk.cc",
        "memset_thunk.cc",
        "outfeed_thunk.cc",
        "replica_id_thunk.cc",
        "sequential_thunk.cc",
        "while_thunk.cc",
    ],
    hdrs = [
        "conditional_thunk.h",
        "convolution_thunk.h",
        "copy_thunk.h",
        "custom_call_thunk.h",
        "for_thunk.h",
        "fused_mha_thunk.h",
        "gemm_thunk.h",
        "gpu_executable.h",
        "infeed_thunk.h",
        "kernel_thunk.h",
        "memset_thunk.h",
        "outfeed_thunk.h",
        "replica_id_thunk.h",
        "sequential_thunk.h",
        "while_thunk.h",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":backend_configs_cc",
        ":buffer_allocations",
        ":cusolver_context",
        ":custom_call_thunk",
        ":fft_thunk",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_executable_run_options",
        ":gpu_fused_mha_runner",
        ":gpu_types",
        ":io_feed_manager",
        ":ir_emission_utils",
        ":kernel_arguments",
        ":launch_dimensions",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":non_atomically_upgradeable_rw_lock",
        ":stream_executor_util",
        ":thunk",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:refcounting_hash_map",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir/runtime/ir:rt",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:type_converter",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/compiler/xla/service/gpu/runtime:executable",
        "//tensorflow/compiler/xla/service/gpu/runtime:support",
        "//tensorflow/compiler/xla/service/gpu/runtime2:executable",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor:kernel",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/lib:scoped_annotation",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ] + if_gpu_is_configured([
        ":cholesky_thunk",
        ":precompiled_kernels",
        ":triangular_solve_thunk",
    ]) + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_stream",
        "//tensorflow/compiler/xla/stream_executor/cuda:cudnn_plugin",
        "//tensorflow/compiler/xla/stream_executor/cuda:cufft_plugin",
        "//tensorflow/compiler/xla/stream_executor/cuda:stream_executor_cuda",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/rocm:stream_executor_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":hlo_traversal",
        ":target_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:ArithDialect",
    ],
)

xla_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/platform:test",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "reduction_utils",
    srcs = ["reduction_utils.cc"],
    hdrs = ["reduction_utils.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ] + if_cuda_is_configured([
        ":gpu_asm_opts_util",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
    ]),
)

cc_library(
    name = "cublas_cudnn",
    srcs = ["cublas_cudnn.cc"],
    hdrs = ["cublas_cudnn.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(ezhulenev): Extract `RunCholesky` into a separate library.
cc_library(
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["//tensorflow/tsl/platform:status"],
)

# TODO(ezhulenev): Extract `RunTriangularSolve` into a separate library.
cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["//tensorflow/tsl/platform:status"],
)

cc_library(
    name = "fft_thunk",
    srcs = ["fft_thunk.cc"],
    hdrs = ["fft_thunk.h"],
    deps = [
        ":buffer_allocations",
        ":thunk",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "gemm_rewriter",
    srcs = ["gemm_rewriter.cc"],
    hdrs = ["gemm_rewriter.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":gpu_types",
        ":ir_emission_utils",
        ":matmul_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/evaluator:hlo_evaluator",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "gemm_rewriter_triton",
    srcs = ["gemm_rewriter_triton.cc"],
    hdrs = ["gemm_rewriter_triton.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_padding_requirements",
        ":gpu_types",
        ":ir_emission_utils",
        ":matmul_utils",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "softmax_rewriter_triton",
    srcs = ["softmax_rewriter_triton.cc"],
    hdrs = ["softmax_rewriter_triton.h"],
    deps = [
        ":backend_configs_cc",
        ":gemm_rewriter_triton",
        ":gpu_types",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gemm_rewriter_triton_test",
    srcs = ["gemm_rewriter_triton_test.cc"],
    deps = [
        ":cublas_padding_requirements",
        ":gemm_rewriter_triton",
        ":gpu_types",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gemm_thunk",
    srcs = ["gemm_thunk.cc"],
    hdrs = ["gemm_thunk.h"],
    deps = [
        ":matmul_utils",
        ":thunk",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:logging",
    ],
)

cc_library(
    name = "cublas_lt_matmul_thunk",
    srcs = if_cuda_is_configured(["cublas_lt_matmul_thunk.cc"]) + if_rocm_is_configured([
        "cublas_lt_matmul_thunk.cc",
    ]),
    hdrs = if_cuda_is_configured(["cublas_lt_matmul_thunk.h"]) + if_rocm_is_configured([
        "cublas_lt_matmul_thunk.h",
    ]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        ":matmul_utils",
        ":thunk",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
    ]) + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_header",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/rocm:hipblas_lt_header",
    ]),
)

cc_library(
    name = "gemm_algorithm_picker",
    srcs = if_cuda_is_configured(["gemm_algorithm_picker.cc"]),
    hdrs = if_cuda_is_configured(["gemm_algorithm_picker.h"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = if_cuda_is_configured([
        ":backend_configs_cc",
        ":buffer_comparator",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_conv_runner",
        ":ir_emission_utils",
        ":matmul_utils",
        ":stream_executor_util",
        ":autotuner_util",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_header",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logger",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/tsl/util/proto:proto_utils",
    ]),
)

cc_library(
    name = "autotuner_util",
    srcs = if_gpu_is_configured(["autotuner_util.cc"]),
    hdrs = if_gpu_is_configured(["autotuner_util.h"]),
    deps = if_gpu_is_configured([
        ":gpu_asm_opts_util",
        ":stream_executor_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:protobuf",
    ]),
)

# We need a separate target, as runtime executable cannot depend on compilation
# pipeline.
cc_library(
    name = "autotuner_compile_util",
    srcs = if_gpu_is_configured(["autotuner_compile_util.cc"]),
    hdrs = if_gpu_is_configured(["autotuner_compile_util.h"]),
    deps = if_gpu_is_configured([
        ":autotuner_util",
        ":gpu_executable_run_options",
        ":ir_emission_utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer_header",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
    ]),
)

xla_cc_test(
    name = "gemm_algorithm_picker_test",
    srcs = ["gemm_algorithm_picker_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":backend_configs_cc",
        ":gemm_algorithm_picker",
        ":gemm_rewriter",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
    ],
)

cc_library(
    name = "matmul_utils",
    srcs = ["matmul_utils.cc"],
    hdrs = ["matmul_utils.h"],
    compatible_with = get_compatible_with_portable(),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_header",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/rocm:hipblas_lt_header",
        "//tensorflow/compiler/xla/stream_executor/rocm:hipblaslt_plugin",
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
    ]) + if_static([
        "//tensorflow/tsl/platform:tensor_float_32_utils",
    ]),
)

xla_cc_test(
    name = "matmul_utils_test",
    srcs = ["matmul_utils_test.cc"],
    deps = [
        ":matmul_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dot_dimension_sorter",
    srcs = ["dot_dimension_sorter.cc"],
    hdrs = ["dot_dimension_sorter.h"],
    deps = [
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "dot_dimension_sorter_test",
    srcs = ["dot_dimension_sorter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":dot_dimension_sorter",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "gpu_async_collective_annotator",
    srcs = ["gpu_async_collective_annotator.cc"],
    hdrs = ["gpu_async_collective_annotator.h"],
    deps = [
        ":backend_configs_cc",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "gpu_async_collective_annotator_test",
    srcs = ["gpu_async_collective_annotator_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_async_collective_annotator",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_convert_async_collectives_to_sync",
    srcs = ["gpu_convert_async_collectives_to_sync.cc"],
    hdrs = ["gpu_convert_async_collectives_to_sync.h"],
    deps = [
        ":backend_configs_cc",
        "//tensorflow/compiler/xla/service:convert_async_collectives_to_sync",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

xla_cc_test(
    name = "gpu_convert_async_collectives_to_sync_test",
    srcs = ["gpu_convert_async_collectives_to_sync_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_convert_async_collectives_to_sync",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "conv_algorithm_picker",
    srcs = if_gpu_is_configured(["conv_algorithm_picker.cc"]),
    hdrs = if_gpu_is_configured(["conv_algorithm_picker.h"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        ":autotuner_util",
        ":backend_configs_cc",
        ":buffer_comparator",
        ":gpu_asm_opts_util",
        ":gpu_autotuning_proto_cc",
        ":gpu_conv_runner",
        ":hlo_algorithm_denylist",
        ":stream_executor_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cudnn_header",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:slow_operation_alarm",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/tsl/platform:logger",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/util/proto:proto_utils",
    ]),
)

xla_cc_test(
    name = "conv_algorithm_picker_test",
    srcs = if_gpu_is_configured(["conv_algorithm_picker_test.cc"]),
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":conv_algorithm_picker",
        ":gpu_conv_rewriter",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "gpu_conv_runner",
    srcs = ["gpu_conv_runner.cc"],
    hdrs = ["gpu_conv_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:lazy_op_runner",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_fused_mha_runner",
    srcs = ["gpu_fused_mha_runner.cc"],
    hdrs = ["gpu_fused_mha_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:lazy_op_runner",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_conv_rewriter",
    srcs = ["gpu_conv_rewriter.cc"],
    hdrs = ["gpu_conv_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "move_copy_to_users",
    srcs = ["move_copy_to_users.cc"],
    hdrs = ["move_copy_to_users.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "move_copy_to_users_test",
    srcs = ["move_copy_to_users_test.cc"],
    deps = [
        ":move_copy_to_users",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_conv_rewriter_test",
    srcs = ["gpu_conv_rewriter_test.cc"],
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_rewriter",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "cusolver_context",
    srcs = if_gpu_is_configured(["cusolver_context.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_context.h"]),
    deps = [
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor/cuda:cusolver_lib",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocblas_wrapper",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocsolver_wrapper",
        "//tensorflow/compiler/xla/stream_executor/rocm:hipsolver_wrapper",
    ]),
)

cc_library(
    name = "cusolver_rewriter",
    srcs = if_gpu_is_configured(["cusolver_rewriter.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_rewriter.h"]),
    deps = if_gpu_is_configured([
        ":cusolver_context",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/algorithm:container",
    ]) + ["//tensorflow/tsl/platform:status"],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:fusion_node_indexing_evaluation",
        "//tensorflow/compiler/xla/service:fusion_queue",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "priority_fusion",
    srcs = ["priority_fusion.cc"],
    hdrs = ["priority_fusion.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:fusion_node_indexing_evaluation",
        "//tensorflow/compiler/xla/service:fusion_queue",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "priority_fusion_test",
    srcs = ["priority_fusion_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_hlo_cost_analysis",
        ":priority_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/ir:hlo_reachability",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "multi_output_fusion_test",
    srcs = ["multi_output_fusion_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        ":multi_output_fusion",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "softmax_rewriter_triton_test",
    srcs = ["softmax_rewriter_triton_test.cc"],
    deps = [
        ":softmax_rewriter_triton",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_sanitize_constant_names",
    srcs = ["gpu_sanitize_constant_names.cc"],
    hdrs = ["gpu_sanitize_constant_names.h"],
    deps = [
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
    ],
)

xla_cc_test(
    name = "gpu_sanitize_constant_names_test",
    srcs = ["gpu_sanitize_constant_names_test.cc"],
    deps = [
        ":gpu_sanitize_constant_names",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "fusion_merger",
    srcs = ["fusion_merger.cc"],
    hdrs = ["fusion_merger.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "fusion_merger_test",
    srcs = ["fusion_merger_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":fusion_merger",
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_conv_padding_legalization",
    srcs = ["gpu_conv_padding_legalization.cc"],
    hdrs = ["gpu_conv_padding_legalization.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shape_inference",
    ],
)

xla_cc_test(
    name = "gpu_conv_padding_legalization_test",
    srcs = ["gpu_conv_padding_legalization_test.cc"],
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_padding_legalization",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "cudnn_support_utils",
    srcs = ["cudnn_support_utils.cc"],
    hdrs = ["cudnn_support_utils.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:status",
    ],
)

xla_cc_test(
    name = "cudnn_support_utils_test",
    srcs = ["cudnn_support_utils_test.cc"],
    deps = [
        ":cudnn_support_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_pad_for_convolutions",
    srcs = ["cudnn_pad_for_convolutions.cc"],
    hdrs = ["cudnn_pad_for_convolutions.h"],
    deps = [
        ":cudnn_support_utils",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

xla_cc_test(
    name = "cudnn_pad_for_convolutions_test",
    srcs = ["cudnn_pad_for_convolutions_test.cc"],
    deps = [
        ":cublas_cudnn",
        ":cudnn_pad_for_convolutions",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "cudnn_vectorize_convolutions",
    srcs = ["cudnn_vectorize_convolutions.cc"],
    hdrs = ["cudnn_vectorize_convolutions.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_support_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ],
)

xla_cc_test(
    name = "cudnn_vectorize_convolutions_test",
    srcs = ["cudnn_vectorize_convolutions_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_vectorize_convolutions",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "cudnn_simplify_padding",
    srcs = ["cudnn_simplify_padding.cc"],
    hdrs = ["cudnn_simplify_padding.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
    ],
)

xla_cc_test(
    name = "cudnn_simplify_padding_test",
    srcs = ["cudnn_simplify_padding_test.cc"],
    deps = [
        ":cublas_cudnn",
        ":cudnn_pad_for_convolutions",
        ":cudnn_simplify_padding",
        ":cudnn_vectorize_convolutions",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "cublas_pad_for_gemms",
    srcs = ["cublas_pad_for_gemms.cc"],
    hdrs = ["cublas_pad_for_gemms.h"],
    deps = [
        ":gemm_rewriter_triton",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

cc_library(
    name = "cublas_padding_requirements",
    srcs = ["cublas_padding_requirements.cc"],
    hdrs = ["cublas_padding_requirements.h"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor:device_description",
    ],
)

xla_cc_test(
    name = "cublas_pad_for_gemms_test",
    srcs = ["cublas_pad_for_gemms_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":cublas_pad_for_gemms",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

tf_proto_library(
    name = "executable_proto",
    srcs = ["executable.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/cleanup",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_reduce_scatter_creator",
    srcs = ["gpu_reduce_scatter_creator.cc"],
    hdrs = ["gpu_reduce_scatter_creator.h"],
    deps = [
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:reduce_scatter_utils",
    ],
)

cc_library(
    name = "gpu_float_support",
    srcs = ["gpu_float_support.cc"],
    hdrs = ["gpu_float_support.h"],
    deps = [
        "//tensorflow/compiler/xla/service:float_support",
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
    deps = [
        ":executable_proto_cc",
        ":fusion_wrapper",
        ":gpu_constants",
        ":gpu_convert_async_collectives_to_sync",
        ":gpu_device_info",
        ":gpu_executable",
        ":gpu_hlo_schedule",
        ":ir_emitter_context",
        ":ir_emitter_unnested",
        ":metrics",
        ":runtime_intrinsics",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir/backends/gpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir/backends/gpu2/transforms:passes",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//tensorflow/compiler/xla/mlir_hlo:transforms_gpu_passes",
        "//tensorflow/compiler/xla/service:bitcast_dtypes_expander",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_rematerialization",
        "//tensorflow/compiler/xla/service:optimization_barrier_expander",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "gpu_compiler",
    srcs = if_gpu_is_configured([
        "gpu_compiler.cc",
    ]),
    hdrs = if_gpu_is_configured([
        "gpu_compiler.h",
    ]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        ":alias_passthrough_params",
        ":all_reduce_blueconnect",
        ":autotuner_util",
        ":compile_module_to_llvm_ir",
        ":conv_layout_normalization",
        ":copy_fusion",
        ":dot_dimension_sorter",
        ":executable_proto_cc",
        ":fusion_merger",
        ":gemm_broadcast_folding_rewriter",
        ":gemm_rewriter_triton",
        ":gemm_rewriter",
        ":gpu_async_collective_annotator",
        ":gpu_constants",
        ":gpu_conv_rewriter",
        ":gpu_cost_model_stats_collection",
        ":gpu_device_info",
        ":gpu_executable",
        ":gpu_float_support",
        ":gpu_hlo_cost_analysis",
        ":gpu_hlo_schedule",
        ":gpu_layout_assignment",
        ":gpu_reduce_scatter_creator",
        ":gpu_sanitize_constant_names",
        ":gpu_scatter_expander",
        ":gpu_shape_verifier",
        ":hlo_fusion_stats",
        ":horizontal_input_fusion",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        ":ir_emission_utils",
        ":ir_emitter",
        ":matmul_utils",
        ":metrics",
        ":move_copy_to_users",
        ":multi_output_fusion",
        ":priority_fusion",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":reduction_splitter",
        ":reduction_utils",
        ":runtime_intrinsics",
        ":scatter_slice_simplifier",
        ":softmax_rewriter_triton",
        ":topk_specializer",
        ":topk_splitter",
        ":tree_reduction_rewriter",
        ":variadic_op_splitter",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/transforms:hlo_constant_splitter",
        "//tensorflow/compiler/xla/mlir/backends/gpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:all_gather_broadcast_reorder",
        "//tensorflow/compiler/xla/service:all_gather_combiner",
        "//tensorflow/compiler/xla/service:all_reduce_combiner",
        "//tensorflow/compiler/xla/service:all_reduce_contiguous",
        "//tensorflow/compiler/xla/service:all_reduce_folder",
        "//tensorflow/compiler/xla/service:all_reduce_promotion",
        "//tensorflow/compiler/xla/service:all_reduce_reassociate",
        "//tensorflow/compiler/xla/service:async_collective_creator",
        "//tensorflow/compiler/xla/service:batchnorm_expander",
        "//tensorflow/compiler/xla/service:bitcast_dtypes_expander",
        "//tensorflow/compiler/xla/service:broadcast_canonicalizer",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:collective_permute_decomposer",
        "//tensorflow/compiler/xla/service:collective_pipeliner",
        "//tensorflow/compiler/xla/service:collectives_schedule_linearizer",
        "//tensorflow/compiler/xla/service:comparison_expander",
        "//tensorflow/compiler/xla/service:conditional_canonicalizer",
        "//tensorflow/compiler/xla/service:conditional_simplifier",
        "//tensorflow/compiler/xla/service:convert_async_collectives_to_sync",
        "//tensorflow/compiler/xla/service:convert_mover",
        "//tensorflow/compiler/xla/service:convolution_4d_expander",
        "//tensorflow/compiler/xla/service:convolution_pred_expander",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:dot_dimension_merger",
        "//tensorflow/compiler/xla/service:dot_merger",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:dynamic_dimension_simplifier",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:dynamic_padder",
        "//tensorflow/compiler/xla/service:eigh_expander",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:float_normalization",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:gather_simplifier",
        "//tensorflow/compiler/xla/service:hlo_computation_deduplicator",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_rematerialization",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:layout_normalization",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:logistic_expander",
        "//tensorflow/compiler/xla/service:loop_schedule_linearizer",
        "//tensorflow/compiler/xla/service:operand_upcaster",
        "//tensorflow/compiler/xla/service:optimization_barrier_expander",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:real_imag_expander",
        "//tensorflow/compiler/xla/service:reduce_decomposer",
        "//tensorflow/compiler/xla/service:reduce_scatter_combiner",
        "//tensorflow/compiler/xla/service:reduce_scatter_reassociate",
        "//tensorflow/compiler/xla/service:reshape_decomposer",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:result_caster",
        "//tensorflow/compiler/xla/service:rng_bit_generator_expander",
        "//tensorflow/compiler/xla/service:rng_expander",
        "//tensorflow/compiler/xla/service:scatter_simplifier",
        "//tensorflow/compiler/xla/service:sharding_propagation",
        "//tensorflow/compiler/xla/service:sharding_remover",
        "//tensorflow/compiler/xla/service:simplify_fp_conversions",
        "//tensorflow/compiler/xla/service:slice_sinker",
        "//tensorflow/compiler/xla/service:slow_operation_alarm",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:stable_sort_expander",
        "//tensorflow/compiler/xla/service:stochastic_convert_decomposer",
        "//tensorflow/compiler/xla/service:topk_rewriter",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_all_reduce_code_motion",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_trip_count_annotator",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/spmd:collective_permute_motion",
        "//tensorflow/compiler/xla/service/spmd:stateful_rng_spmd_partitioner",
        "//tensorflow/compiler/xla/stream_executor:device_description_proto_cc_impl",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla",
        "//tensorflow/tsl/platform:blocking_counter",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/profiler/lib:traceme",
    ]) + ["//tensorflow/tsl/lib/monitoring:counter"],
)

xla_cc_test(
    name = "gpu_compiler_test",
    srcs = ["gpu_compiler_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":horizontal_loop_fusion",
        ":metrics",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "auto_sharding_gpu_compiler_test",
    srcs = ["auto_sharding_gpu_compiler_test.cc"],
    tags = tf_cuda_tests_tags() + ["no_oss"],  # TODO(b/277355322): Make autosharding work in OSS
    deps = [
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "nvptx_compiler",
    srcs = if_cuda_is_configured([
        "nvptx_compiler_registration.cc",
    ]),
    deps = if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        ":nvptx_compiler_impl",
        "//tensorflow/tsl/platform:path",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "nvptx_compiler_impl",
    srcs = if_cuda_is_configured([
        "nvptx_compiler.cc",
    ]),
    hdrs = if_cuda_is_configured([
        "nvptx_compiler.h",
    ]),
    deps = if_cuda_is_configured([
        ":autotuner_util",
        ":cublas_cudnn",
        ":cublas_pad_for_gemms",
        ":cublas_padding_requirements",
        ":cudnn_fused_conv_rewriter",
        ":cudnn_fused_mha_rewriter",
        ":cudnn_fused_mha_transpose_fusion",
        ":cudnn_pad_for_convolutions",
        ":cudnn_simplify_padding",
        ":cudnn_vectorize_convolutions",
        ":cusolver_rewriter",
        ":gemm_algorithm_picker",
        ":gpu_asm_opts_util",
        ":gpu_compiler",
        ":conv_algorithm_picker",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_executable",
        ":gpu_layout_assignment",
        ":ir_emission_utils",
        ":metrics",
        ":target_constants",
        ":triangular_solve_rewriter",
        ":triton_autotuner",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:convert_mover",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:float_normalization",
        "//tensorflow/compiler/xla/service:float_support",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_diagnostics",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/util:env_var",
        "//tensorflow/compiler/xla/service:layout_normalization",
        "//tensorflow/compiler/xla/service:reshape_decomposer",
    ]),
)

xla_cc_test(
    name = "nvptx_compiler_test",
    srcs = if_gpu_is_configured([
        "nvptx_compiler_test.cc",
    ]),
    tags = tf_cuda_tests_tags() + [
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false
        # positives in msan.
    ],
    deps = [
        ":nvptx_compiler_impl",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

xla_cc_test(
    name = "gpu_aot_compilation_test",
    srcs = if_cuda_is_configured([
        "gpu_aot_compilation_test.cc",
    ]),
    env = {
        "XLA_FLAGS": "--xla_gpu_enable_xla_runtime_executable",
    },
    tags = [
        "gpu",
        "no_oss",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
        "requires-gpu-nvidia",
    ],
    deps = if_cuda_is_configured([
        ":nvptx_compiler_impl",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ]),
)

cc_library(
    name = "amdgpu_compiler",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler_registration.cc",
    ]),
    deps = if_rocm_is_configured([
        ":amdgpu_compiler_impl",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "amdgpu_compiler_impl",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler.cc",
    ]),
    hdrs = if_rocm_is_configured([
        "amdgpu_compiler.h",
    ]),
    deps = if_rocm_is_configured([
        ":autotuner_util",
        ":cusolver_rewriter",
        ":gemm_rewriter",
        ":gpu_compiler",
        ":conv_algorithm_picker",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_layout_assignment",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":target_constants",
        ":tree_reduction_rewriter",
        ":triangular_solve_rewriter",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/tsl/platform:rocm_rocdl_path",
    ]),
)

cc_library(
    name = "all_reduce_blueconnect",
    srcs = ["all_reduce_blueconnect.cc"],
    hdrs = ["all_reduce_blueconnect.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "all_reduce_blueconnect_test",
    srcs = ["all_reduce_blueconnect_test.cc"],
    deps = [
        ":all_reduce_blueconnect",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "xfeed_queue",
    hdrs = ["xfeed_queue.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "io_feed_manager",
    srcs = [
        "infeed_manager.cc",
        "outfeed_manager.cc",
        "xla_executor_state.h",
    ],
    hdrs = [
        "infeed_manager.h",
        "outfeed_manager.h",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":xfeed_queue",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:notification",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_shape_verifier",
    srcs = ["gpu_shape_verifier.cc"],
    hdrs = ["gpu_shape_verifier.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_verifier",
    ],
)

cc_library(
    name = "gpu_layout_assignment",
    srcs = ["gpu_layout_assignment.cc"],
    hdrs = ["gpu_layout_assignment.h"],
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        ":matmul_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gpu_layout_assignment_test",
    srcs = ["gpu_layout_assignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gemm_rewriter",
        ":gpu_layout_assignment",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_schedule",
    srcs = ["gpu_hlo_schedule.cc"],
    hdrs = ["gpu_hlo_schedule.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":gpu_device_info",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:latency_hiding_scheduler",
        "//tensorflow/compiler/xla/service:latency_hiding_scheduler_preparation",
        "//tensorflow/compiler/xla/service:profile_guided_latency_estimator",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "gpu_hlo_schedule_test",
    srcs = [
        "gpu_hlo_schedule_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_device_info",
        ":gpu_hlo_schedule",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

xla_cc_test(
    name = "while_transformer_test",
    srcs = ["while_transformer_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":instruction_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "stream_executor_util",
    srcs = ["stream_executor_util.cc"],
    hdrs = ["stream_executor_util.h"],
    copts = tsl_copts(),
    deps = [
        ":cublas_cudnn",
        ":launch_dimensions",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:kernel_spec",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/util:env_var",
        "//tensorflow/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_asm_opts_util",
    srcs = ["gpu_asm_opts_util.cc"],
    hdrs = ["gpu_asm_opts_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tsl_copts(),
    deps = [
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    srcs = ["gpu_hlo_cost_analysis.cc"],
    hdrs = [
        "gpu_hlo_cost_analysis.h",
        "hlo_op_profiles.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":gpu_device_info",
        ":hlo_op_profile_proto_cc",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_hlo_cost_analysis_test",
    srcs = ["gpu_hlo_cost_analysis_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_fusion_analysis",
    srcs = ["hlo_fusion_analysis.cc"],
    hdrs = [
        "hlo_fusion_analysis.h",
        "kernel_mapping_scheme.h",
    ],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":gpu_device_info",
        ":gpu_fusible",
        ":hlo_traversal",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":reduction_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:union_find",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_query",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "//tensorflow/tsl/platform:macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "gpu_performance_model",
    srcs = ["gpu_performance_model.cc"],
    hdrs = ["gpu_performance_model.h"],
    deps = [
        ":backend_configs_cc",
        ":gpu_hlo_cost_analysis",
        ":hlo_fusion_analysis",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "gpu_performance_model_test",
    srcs = ["gpu_performance_model_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_device_info",
        ":gpu_device_info_for_tests",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_proto_library(
    name = "hlo_op_profile_proto",
    srcs = ["hlo_op_profile.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_library(
    name = "hlo_op_profiler_lib",
    testonly = True,
    srcs = ["hlo_op_profiler.cc"],
    hdrs = ["hlo_op_profiler.h"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    deps = [
        ":gpu_device_info",
        ":hlo_op_profile_proto_cc",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/backends/profiler/gpu:cupti_collector",
        "//tensorflow/compiler/xla/backends/profiler/gpu:cupti_tracer",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service:interpreter_plugin",
        "//tensorflow/compiler/xla/tests:test_utils",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "hlo_op_profiler_run",
    timeout = "eternal",
    srcs = ["hlo_op_profiler_run.cc"],
    # Disable backend optimizations (in particular reassociate and instcombine) which would optimize
    # expressions like integer add and multiply.
    args = ["--xla_backend_optimization_level=0"],
    # This is a development tool, not a normal test, and thus should only be run
    # manually with --config=cuda.
    tags = [
        "gpu",
        "manual",
        "notap",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":gpu_device_info",
        ":hlo_op_profile_proto_cc",
        ":hlo_op_profiler_lib",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/util:command_line_flags",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "hlo_op_profiler_test",
    srcs = ["hlo_op_profiler_test.cc"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":hlo_op_profiler_lib",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "buffer_comparator",
    srcs = if_cuda_is_configured(["buffer_comparator.cc"]),
    hdrs = if_cuda_is_configured(["buffer_comparator.h"]),
    deps = if_cuda_is_configured([
        ":launch_dimensions",
        ":gpu_asm_opts_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/tsl/platform:statusor",
    ]),
)

xla_cc_test(
    name = "buffer_comparator_test",
    srcs = if_cuda_is_configured(["buffer_comparator_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ] + if_cuda_is_configured([
        ":buffer_comparator",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
    ]),
)

cc_library(
    name = "gpu_fusible",
    srcs = ["gpu_fusible.cc"],
    hdrs = ["gpu_fusible.h"],
    deps = [
        ":gpu_device_info",
        ":ir_emission_utils",
        ":reduction_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "gpu_fusible_test",
    srcs = ["gpu_fusible_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":gpu_fusible",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_fused_conv_rewriter",
    srcs = ["cudnn_fused_conv_rewriter.cc"],
    hdrs = ["cudnn_fused_conv_rewriter.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

xla_cc_test(
    name = "cudnn_fused_conv_rewriter_test",
    srcs = ["cudnn_fused_conv_rewriter_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    shard_count = 10,
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        # This test runs some fusions that are only supported on Ampere+.
        "requires-gpu-sm80",
    ],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_fused_conv_rewriter",
        ":gpu_conv_rewriter",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:convert_mover",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

cc_library(
    name = "cudnn_fused_mha_rewriter",
    srcs = ["cudnn_fused_mha_rewriter.cc"],
    hdrs = ["cudnn_fused_mha_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":ir_emission_utils",
        ":matmul_utils",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cudnn_fused_mha_transpose_fusion",
    srcs = ["cudnn_fused_mha_transpose_fusion.cc"],
    hdrs = ["cudnn_fused_mha_transpose_fusion.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":matmul_utils",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "cudnn_fused_mha_rewriter_test",
    srcs = ["cudnn_fused_mha_rewriter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_fused_mha_rewriter",
        ":cudnn_fused_mha_transpose_fusion",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:layout_normalization",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service:reshape_decomposer",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "float_support_test",
    srcs = ["float_support_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-sm80",
    ]},
    backends = [
        "gpu",
    ],
    deps = [
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "conv_layout_normalization_test",
    srcs = ["conv_layout_normalization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "variadic_op_splitter",
    srcs = ["variadic_op_splitter.cc"],
    hdrs = ["variadic_op_splitter.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_scatter_expander",
    srcs = ["gpu_scatter_expander.cc"],
    hdrs = ["gpu_scatter_expander.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "variadic_op_splitter_test",
    srcs = ["variadic_op_splitter_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":ir_emission_utils",
        ":variadic_op_splitter",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_proto_library(
    name = "gpu_autotuning_proto",
    srcs = ["gpu_autotuning.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xla:autotuning_proto",
    ],
)

cc_library(
    name = "hlo_algorithm_denylist",
    srcs = ["hlo_algorithm_denylist.cc"],
    hdrs = ["hlo_algorithm_denylist.h"],
    deps = [
        ":gpu_autotuning_proto_cc",
        "//tensorflow/compiler/xla:autotuning_proto_cc",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

xla_cc_test(
    name = "hlo_algorithm_denylist_test",
    srcs = ["hlo_algorithm_denylist_test.cc"],
    data = ["data/hlo_algorithm_denylist.pbtxt"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":hlo_algorithm_denylist",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:resource_loader",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "alias_passthrough_params",
    srcs = ["alias_passthrough_params.cc"],
    hdrs = ["alias_passthrough_params.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "alias_passthrough_params_test",
    srcs = ["alias_passthrough_params_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":alias_passthrough_params",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "horizontal_loop_fusion",
    srcs = ["horizontal_loop_fusion.cc"],
    hdrs = ["horizontal_loop_fusion.h"],
    deps = [
        ":gpu_fusible",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "horizontal_loop_fusion_test",
    srcs = ["horizontal_loop_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_device_info_for_tests",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "horizontal_input_fusion",
    srcs = ["horizontal_input_fusion.cc"],
    hdrs = ["horizontal_input_fusion.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "horizontal_input_fusion_test",
    srcs = ["horizontal_input_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_device_info_for_tests",
        ":horizontal_input_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_degenerate_dim_remover",
    srcs = ["reduction_degenerate_dim_remover.cc"],
    hdrs = ["reduction_degenerate_dim_remover.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduction_dimension_grouper",
    srcs = ["reduction_dimension_grouper.cc"],
    hdrs = ["reduction_dimension_grouper.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "reduction_splitter",
    srcs = ["reduction_splitter.cc"],
    hdrs = ["reduction_splitter.h"],
    deps = [
        ":reduction_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "reduction_splitter_test",
    srcs = ["reduction_splitter_test.cc"],
    deps = [
        ":reduction_splitter",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_layout_normalizer",
    srcs = ["reduction_layout_normalizer.cc"],
    hdrs = ["reduction_layout_normalizer.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["tree_reduction_rewriter.cc"],
    hdrs = ["tree_reduction_rewriter.h"],
    deps = [
        ":gpu_types",
        ":reduction_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gemm_broadcast_folding_rewriter",
    srcs = ["gemm_broadcast_folding_rewriter.cc"],
    hdrs = ["gemm_broadcast_folding_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
    ],
)

# See tap/tensorflow.xla_gpu_jitrt.
test_suite(
    name = "jitrt_executable_tests",
    tests = [
        ":cudnn_fused_conv_rewriter_test",
        ":cudnn_fused_mha_rewriter_test",
        ":custom_call_test",
        # copybara:uncomment ":gpu_aot_compilation_test",
        # copybara:uncomment "//platforms/xla/tests/internal:xfeed_test_gpu",
        # copybara:uncomment "//third_party/py/jax/experimental/jax2tf/tests:primitives_test_gpu",
        # copybara:uncomment "//third_party/py/jax/tests:pmap_test_gpu",
        # copybara:uncomment "//tensorflow/compiler/tests:fft_test_gpu",
        "//tensorflow/compiler/xla/python:xla_client_test_gpu",
        "//tensorflow/compiler/xla/service/gpu/tests:add_preds.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:concat.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:constant.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy_nested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:dynamic_update_slice_inplace.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice_different_operands.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fusion.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:gemm_broadcast_folding_rewrite_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gemm_rewrite_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_alignment_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_atomic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_compilation_parallelism_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_convolution_regression_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_alone_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_dyn_shape_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ftz_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_fusion_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_index_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_infeed_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_input_fusible_slice_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_kernel_tiling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ldg_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_noalias_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_reduce_scatter_creator_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_spmd_e2e_compile_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_too_many_blocks_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_unrolling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:in_place_op_test",
        "//tensorflow/compiler/xla/service/gpu/tests:kernel_launch_test",
        "//tensorflow/compiler/xla/service/gpu/tests:kernel_reuse.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:launch_dimensions.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:pad_to_static.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:parallel_reduction_test",
        "//tensorflow/compiler/xla/service/gpu/tests:pred_arithmetic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduce_unnested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_degenerate_dim_remover_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_dimension_grouper_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_layout_normalizer_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_sm_all.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:rng_get_and_update_state.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:select_and_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:select_and_scatter_test",
        "//tensorflow/compiler/xla/service/gpu/tests:single_instruction.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:slice_to_dynamic.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting_test",
        "//tensorflow/compiler/xla/service/gpu/tests:swap_conv_operands_test",
        "//tensorflow/compiler/xla/service/gpu/tests:tree_reduction_rewriter_test",
        "//tensorflow/compiler/xla/tests:all_reduce_test_gpu",
        "//tensorflow/compiler/xla/tests:array_elementwise_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:axpy_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:bad_rng_shape_validation_test_gpu",
        "//tensorflow/compiler/xla/tests:batch_normalization_test_gpu",
        "//tensorflow/compiler/xla/tests:bfloat16_test_gpu",
        "//tensorflow/compiler/xla/tests:binop_scaling_test_gpu",
        "//tensorflow/compiler/xla/tests:bitcast_convert_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_test_gpu",
        "//tensorflow/compiler/xla/tests:buffer_donation_test_gpu",
        "//tensorflow/compiler/xla/tests:call_test_gpu",
        "//tensorflow/compiler/xla/tests:check_execution_arity_test_gpu",
        "//tensorflow/compiler/xla/tests:cholesky_test_gpu",
        "//tensorflow/compiler/xla/tests:client_test_gpu",
        "//tensorflow/compiler/xla/tests:compilation_cache_test_gpu",
        "//tensorflow/compiler/xla/tests:compute_constant_test_gpu",
        "//tensorflow/compiler/xla/tests:concat_test_gpu",
        "//tensorflow/compiler/xla/tests:conditional_test_gpu",
        "//tensorflow/compiler/xla/tests:constant_reduction_function_test_gpu",
        "//tensorflow/compiler/xla/tests:constants_test_gpu",
        "//tensorflow/compiler/xla/tests:conv_depthwise_backprop_filter_test_gpu",
        "//tensorflow/compiler/xla/tests:conv_depthwise_test_gpu",
        "//tensorflow/compiler/xla/tests:convert_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_dimension_numbers_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_no_vmodule_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_cudnn_frontend_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_variants_test_gpu",
        "//tensorflow/compiler/xla/tests:copy_test_gpu",
        "//tensorflow/compiler/xla/tests:cpu_gpu_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:deallocation_test_gpu",
        "//tensorflow/compiler/xla/tests:deconstruct_tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:deep_graph_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_single_threaded_runtime_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:dynamic_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:float8_test_gpu",
        "//tensorflow/compiler/xla/tests:floor_ceil_test_gpu",
        "//tensorflow/compiler/xla/tests:fmax_fmin_test_gpu",
        "//tensorflow/compiler/xla/tests:gather_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:get_dimension_size_test_gpu",
        "//tensorflow/compiler/xla/tests:grouped_convolution_test_gpu",
        "//tensorflow/compiler/xla/tests:half_test_gpu",
        "//tensorflow/compiler/xla/tests:iota_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_allocation_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_execute_test_gpu",
        "//tensorflow/compiler/xla/tests:log_test_gpu",
        "//tensorflow/compiler/xla/tests:map_test_gpu",
        "//tensorflow/compiler/xla/tests:matmul_test_gpu",
        "//tensorflow/compiler/xla/tests:matrix_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:multidimensional_slice_test_gpu",
        "//tensorflow/compiler/xla/tests:multioutput_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:outfeed_in_nested_computation_test_gpu",
        "//tensorflow/compiler/xla/tests:pad_test_gpu",
        "//tensorflow/compiler/xla/tests:params_test_gpu",
        "//tensorflow/compiler/xla/tests:pred_test_gpu",
        "//tensorflow/compiler/xla/tests:prng_test_gpu",
        "//tensorflow/compiler/xla/tests:ptxas_bug_120501638_gpu",
        "//tensorflow/compiler/xla/tests:query_inferred_shape_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_hlo_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_precision_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_window_test_gpu",
        "//tensorflow/compiler/xla/tests:replay_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_motion_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_test_gpu",
        "//tensorflow/compiler/xla/tests:reverse_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_packed_literal_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_transfer_test_gpu",
        "//tensorflow/compiler/xla/tests:sample_text_test_gpu",
        "//tensorflow/compiler/xla/tests:scalar_computations_test_gpu",
        "//tensorflow/compiler/xla/tests:scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:select_and_scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:select_test_gpu",
        "//tensorflow/compiler/xla/tests:slice_test_gpu",
        "//tensorflow/compiler/xla/tests:token_hlo_test_gpu",
        "//tensorflow/compiler/xla/tests:transfer_manager_test_gpu",
        "//tensorflow/compiler/xla/tests:transpose_test_gpu",
        "//tensorflow/compiler/xla/tests:triangular_solve_test_gpu",
        "//tensorflow/compiler/xla/tests:tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:unary_op_test_gpu",
        "//tensorflow/compiler/xla/tests:value_inference_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_reduce_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:while_test_gpu",
        "//tensorflow/compiler/xla/tests:xla_hlo_profile_test_gpu",
    ] + if_google([
        # Currently fails in OSS.
        "//tensorflow/python/kernel_tests/signal:fft_ops_test_xla_gpu",
    ]),
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/tsl/lib/monitoring:counter",
        "//tensorflow/tsl/lib/monitoring:sampler",
    ],
)

cc_library(
    name = "precompiled_kernels",
    srcs = if_gpu_is_configured(["precompiled_kernels.cc"]),
    hdrs = if_gpu_is_configured(["precompiled_kernels.h"]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_helpers",
    ]),
)

cc_library(
    name = "triangular_solve_rewriter",
    srcs = ["triangular_solve_rewriter.cc"],
    hdrs = ["triangular_solve_rewriter.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
    ],
)

tsl_gpu_library(
    name = "runtime_intrinsics",
    srcs = ["runtime_intrinsics.cc"],
    hdrs = ["runtime_intrinsics.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "runtime_intrinsics_test",
    srcs = ["runtime_intrinsics_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":runtime_intrinsics",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "hlo_fusion_stats",
    srcs = ["hlo_fusion_stats.cc"],
    hdrs = ["hlo_fusion_stats.h"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_fusion_stats_test",
    srcs = ["hlo_fusion_stats_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":hlo_fusion_stats",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "scatter_slice_simplifier",
    srcs = ["scatter_slice_simplifier.cc"],
    hdrs = ["scatter_slice_simplifier.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "scatter_slice_simplifier_test",
    srcs = ["scatter_slice_simplifier_test.cc"],
    deps = [
        ":scatter_slice_simplifier",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "conv_layout_normalization",
    srcs = ["conv_layout_normalization.cc"],
    hdrs = ["conv_layout_normalization.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "topk_specializer",
    srcs = if_cuda_is_configured(
        ["topk_specializer.cc"],
        ["topk_specializer_nocuda.cc"],
    ),
    hdrs = ["topk_specializer.h"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:tuple_util",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "topk_splitter",
    srcs = ["topk_splitter.cc"],
    hdrs = ["topk_splitter.h"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "topk_splitter_test",
    srcs = ["topk_splitter_test.cc"],
    deps = [
        ":topk_splitter",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:topk_rewriter",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "copy_fusion",
    srcs = ["copy_fusion.cc"],
    hdrs = ["copy_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":ir_emission_utils",
        ":reduction_utils",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "kernel_reuse_cache",
    srcs = ["kernel_reuse_cache.cc"],
    hdrs = ["kernel_reuse_cache.h"],
    deps = [
        ":kernel_arguments",
        ":launch_dimensions",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_arguments",
    srcs = ["kernel_arguments.cc"],
    hdrs = ["kernel_arguments.h"],
    deps = [
        ":gpu_constants",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:transforms_gpu_passes",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "hlo_traversal",
    srcs = ["hlo_traversal.cc"],
    hdrs = ["hlo_traversal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "hlo_traversal_test",
    srcs = ["hlo_traversal_test.cc"],
    deps = [
        ":hlo_traversal",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fusion_wrapper",
    srcs = ["fusion_wrapper.cc"],
    hdrs = ["fusion_wrapper.h"],
    deps = [
        ":gpu_fusible",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "fusion_wrapper_test",
    srcs = ["fusion_wrapper_test.cc"],
    deps = [
        ":fusion_wrapper",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "copy_fusion_test",
    srcs = ["copy_fusion_test.cc"],
    deps = [
        ":copy_fusion",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "autotuner_util_test",
    srcs = if_cuda_is_configured(["autotuner_util_test.cc"]),
    deps = if_cuda_is_configured([
        ":autotuner_util",
        "//tensorflow/compiler/xla:autotune_results_proto_cc",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:protobuf",
    ]) + ["//tensorflow/compiler/xla/tests:xla_internal_test_main"],
)

cc_library(
    name = "gpu_cost_model_stats_collection",
    srcs = ["gpu_cost_model_stats_collection.cc"],
    hdrs = ["gpu_cost_model_stats_collection.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_cost_model_stats_collection_test",
    srcs = ["gpu_cost_model_stats_collection_test.cc"],
    deps = [
        ":backend_configs_cc",
        ":gpu_cost_model_stats_collection",
        ":gpu_device_info_for_tests",
        ":gpu_hlo_cost_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
    ],
)
