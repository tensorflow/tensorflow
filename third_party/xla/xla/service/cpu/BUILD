# Description:
#    LLVM-based CPU backend for XLA.

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//third_party/compute_library:build_defs.bzl",
    "acl_deps",
    "if_enable_acl",
)
load(
    "//xla:xla.bzl",
    "ORC_JIT_MEMORY_MAPPER_TARGETS",
    "xla_cc_binary",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility", "tf_openmp_copts", "tsl_copts")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//xla/tsl/mkl:build_defs.bzl",
    "mkl_deps",
)
load(":build_defs.bzl", "runtime_copts")

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

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

bool_flag(
    name = "experimental_mlir_gpu",
    build_setting_default = False,
)

config_setting(
    name = "experimental_mlir_gpu_enabled",
    flag_values = {
        ":experimental_mlir_gpu": "True",
    },
)

cc_library(
    name = "test_header_helper",
    testonly = True,
    hdrs = ["test_target_triple_helper.h"],
)

# When using mlir based HloLowering, the following utils will sometimes be needed to define used symbols.
cc_library(
    name = "runtime_mlir_utils",
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)

filegroup(
    name = "runtime_srcs",
    srcs = [
        # Single-threaded support.
        "runtime_custom_call_status.cc",
        "runtime_fp16.cc",
        "runtime_key_value_sort.cc",
        "runtime_pow.cc",
        "runtime_single_threaded_conv2d.cc",
        "runtime_single_threaded_conv3d.cc",
        "runtime_single_threaded_fft.cc",
        "runtime_single_threaded_matmul_c128.cc",
        "runtime_single_threaded_matmul_c64.cc",
        "runtime_single_threaded_matmul_common.h",
        "runtime_single_threaded_matmul_f16.cc",
        "runtime_single_threaded_matmul_f32.cc",
        "runtime_single_threaded_matmul_f64.cc",
        "runtime_single_threaded_matmul_s32.cc",
        "runtime_topk.cc",
        # Multi-threaded support.
        "runtime_conv2d.cc",
        "runtime_conv3d.cc",
        "runtime_fft.cc",
        "runtime_matmul_c128.cc",
        "runtime_matmul_c64.cc",
        "runtime_matmul_common.h",
        "runtime_matmul_f16.cc",
        "runtime_matmul_f32.cc",
        "runtime_matmul_f64.cc",
        "runtime_matmul_s32.cc",
        "runtime_fork_join.cc",
        #"runtime_handle_ffi_call.cc", # TODO(b/338344732): Add  "runtime_handle_ffi_call.cc".
    ],
    visibility = internal_visibility([":friends"]),
)

filegroup(
    name = "runtime_hdrs",
    srcs = [
        # XLA Runtime support.
        "buffer_desc.h",
        # Single-threaded support.
        "runtime_custom_call_status.h",
        "runtime_conv_impl.h",
        "runtime_fp16.h",
        "runtime_key_value_sort.h",
        "runtime_pow.h",
        "runtime_single_threaded_conv2d.h",
        "runtime_single_threaded_conv3d.h",
        "runtime_single_threaded_fft.h",
        "runtime_single_threaded_matmul.h",
        "runtime_topk.h",
        # Multi-threaded support.
        "runtime_conv2d.h",
        "runtime_conv3d.h",
        "runtime_fft.h",
        "runtime_fork_join.h",
        "runtime_lightweight_check.h",
        "runtime_matmul.h",
        #"runtime_handle_ffi_call.h", # TODO(b/338344732): Add  "runtime_handle_ffi_call.h"
    ],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "cpu_event",
    hdrs = ["cpu_event.h"],
)

cc_library(
    name = "cpu_xfeed",
    srcs = ["cpu_xfeed.cc"],
    hdrs = ["cpu_xfeed.h"],
    deps = [
        ":cpu_runtime",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:shaped_buffer",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:notification",
    ],
)

cc_library(
    name = "cpu_transfer_manager",
    srcs = ["cpu_transfer_manager.cc"],
    hdrs = ["cpu_transfer_manager.h"],
    deps = [
        ":cpu_runtime",
        ":cpu_xfeed",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:compiler",
        "//xla/service:generic_transfer_manager",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/host:host_platform_id",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "buffer_info_util",
    srcs = ["buffer_info_util.cc"],
    hdrs = ["buffer_info_util.h"],
    deps = [
        "//xla:cpu_function_runtime",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cpu_compiler_pure",
    srcs = ["cpu_compiler.cc"],
    hdrs = ["cpu_compiler.h"],
    copts = tsl_copts(),
    deps = [
        ":buffer_info_util",
        ":compiler_functor",
        ":conv_canonicalization",
        ":cpu_executable",
        ":cpu_float_support",
        ":cpu_instruction_fusion",
        ":cpu_layout_assignment",
        ":cpu_options",
        ":dot_op_emitter",
        ":executable_proto_cc",
        ":ir_emission_utils",
        ":ir_emitter",
        ":onednn_matmul_rewriter",
        ":onednn_ops_rewriter",
        ":parallel_task_assignment",
        ":simple_orc_jit",
        ":target_machine_features",
        ":xla_framework",
        "//xla:cpu_function_runtime",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:all_passes",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:transforms_passes",
        "//xla/service:algebraic_simplifier",
        "//xla/service:all_reduce_promotion",
        "//xla/service:all_to_all_decomposer",
        "//xla/service:batch_dot_simplification",
        "//xla/service:batchnorm_expander",
        "//xla/service:bitcast_dtypes_expander",
        "//xla/service:broadcast_canonicalizer",
        "//xla/service:buffer_assignment",
        "//xla/service:call_graph",
        "//xla/service:call_inliner",
        "//xla/service:change_op_data_type",
        "//xla/service:cholesky_expander",
        "//xla/service:comparison_expander",
        "//xla/service:compiler",
        "//xla/service:conditional_canonicalizer",
        "//xla/service:conditional_simplifier",
        "//xla/service:conditional_to_select",
        "//xla/service:convolution_group_converter",
        "//xla/service:copy_insertion",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:dot_decomposer",
        "//xla/service:dump",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:dynamic_dimension_simplifier",
        "//xla/service:dynamic_index_splitter",
        "//xla/service:dynamic_padder",
        "//xla/service:eigh_expander",
        "//xla/service:executable",
        "//xla/service:flatten_call_graph",
        "//xla/service:float_normalization",
        "//xla/service:float_support",
        "//xla/service:gather_expander",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_memory_scheduler",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_profile_printer_data_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "//xla/service:hlo_verifier",
        "//xla/service:indexed_array_analysis",
        "//xla/service:layout_assignment",
        "//xla/service:llvm_compiler",
        "//xla/service:logical_buffer",
        "//xla/service:logistic_expander",
        "//xla/service:map_inliner",
        "//xla/service:operand_upcaster",
        "//xla/service:optimization_barrier_expander",
        "//xla/service:optimize_input_output_buffer_alias",
        "//xla/service:qr_expander",
        "//xla/service:reduce_decomposer",
        "//xla/service:reshape_decomposer",
        "//xla/service:reshape_mover",
        "//xla/service:result_caster",
        "//xla/service:rng_bit_generator_expander",
        "//xla/service:rng_expander",
        "//xla/service:scatter_expander",
        "//xla/service:select_and_scatter_expander",
        "//xla/service:sharding_propagation",
        "//xla/service:sharding_remover",
        "//xla/service:simplify_fp_conversions",
        "//xla/service:slice_sinker",
        "//xla/service:slow_operation_alarm",
        "//xla/service:sort_simplifier",
        "//xla/service:stochastic_convert_decomposer",
        "//xla/service:sub_byte_normalization",
        "//xla/service:topk_rewriter",
        "//xla/service:transpose_folding",
        "//xla/service:tree_reduction_rewriter",
        "//xla/service:triangular_solve_expander",
        "//xla/service:tuple_simplifier",
        "//xla/service:while_loop_constant_sinking",
        "//xla/service:while_loop_invariant_code_motion",
        "//xla/service:while_loop_simplifier",
        "//xla/service:zero_sized_hlo_elimination",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]),
)

cc_library(
    # The old target name will still be used so that dependencies won't break.
    # In the future, dependencies should be cleaned up and relinked to the above
    # target if registration is not necesary.
    name = "cpu_compiler",
    srcs = ["cpu_compiler_registerer.cc"],
    hdrs = ["cpu_compiler.h"],
    deps = [
        "cpu_compiler_pure",
        ":executable_proto_cc",
        ":target_machine_features",
        ":xla_framework",
        "//xla:cpu_function_runtime",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:buffer_assignment",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_profile_printer_data_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform_id",
        "@llvm-project//llvm:Target",
    ],
    alwayslink = True,  # Contains compiler registration
)

tf_proto_library(
    name = "executable_proto",
    srcs = ["executable.proto"],
    cc_api_version = 2,
    protodeps = [
        ":xla_framework_proto",
        "//xla/service:hlo_proto",
        "//xla:xla_proto",
    ],
)

tf_proto_library(
    name = "xla_framework_proto",
    srcs = ["xla_framework.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "xla_framework",
    hdrs = ["xla_framework.h"],
    deps = [":xla_framework_proto_cc"],
)

cc_library(
    name = "simple_orc_jit",
    srcs = [
        "simple_orc_jit.cc",
        "windows_compatibility.cc",
        "windows_compatibility.h",
    ],
    hdrs = ["simple_orc_jit.h"],
    copts = if_enable_acl(["-DXLA_CPU_USE_ACL=1"]) + tsl_copts(),
    deps = [
        ":compiler_functor",
        ":cpu_runtime",
        ":onednn_layer_norm",
        ":onednn_matmul",
        ":onednn_softmax",
        ":orc_jit_memory_mapper",
        ":runtime_conv2d",
        ":runtime_conv2d_acl",
        ":runtime_conv2d_mkl",
        ":runtime_conv3d",
        ":runtime_custom_call_status",
        ":runtime_fft",
        ":runtime_fork_join",
        ":runtime_fp16",
        ":runtime_handle_ffi_call",
        ":runtime_key_value_sort",
        ":runtime_matmul",
        ":runtime_matmul_acl",
        ":runtime_pow",
        ":runtime_single_threaded_conv2d",
        ":runtime_single_threaded_conv3d",
        ":runtime_single_threaded_fft",
        ":runtime_single_threaded_matmul",
        ":runtime_topk",
        "//xla:types",
        "//xla:util",
        "//xla/service:custom_call_target_registry",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",  # fixdeps: keep
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",  # fixdeps: keep
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@local_tsl//tsl/platform:logging",
    ] + ORC_JIT_MEMORY_MAPPER_TARGETS,
)

cc_library(
    name = "runtime_lightweight_check",
    hdrs = ["runtime_lightweight_check.h"],
    compatible_with = get_compatible_with_portable(),
    copts = runtime_copts(),
)

cc_library(
    name = "runtime_fp16",
    srcs = [
        "runtime_fp16.cc",
    ],
    hdrs = [
        "runtime_fp16.h",
    ],
    copts = runtime_copts(),
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_library(
    name = "runtime_pow",
    srcs = [
        "runtime_pow.cc",
    ],
    hdrs = [
        "runtime_pow.h",
    ],
    copts = runtime_copts(),
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_library(
    name = "buffer_desc",
    hdrs = ["buffer_desc.h"],
)

cc_library(
    name = "cpu_executable",
    srcs = ["cpu_executable.cc"],
    hdrs = ["cpu_executable.h"],
    deps = [
        ":buffer_desc",
        ":simple_orc_jit",
        ":xla_framework",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:computation_layout",
        "//xla/service:custom_call_status_internal",
        "//xla/service:executable",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_execution_profile",
        "//xla/service:logical_buffer",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:xla_debug_info_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor/host:host_stream",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ir_emitter",
    srcs = [
        "elemental_ir_emitter.cc",
        "ir_emitter.cc",
    ],
    hdrs = [
        "elemental_ir_emitter.h",
        "ir_emitter.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":backend_config_proto_cc",
        ":cpu_options",
        ":cpu_runtime",
        ":dot_op_emitter",
        ":ir_emission_utils",
        ":ir_function",
        ":onednn_memory_util",
        ":parallel_loop_emitter",
        ":target_machine_features",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:hlo_module_config",
        "//xla/service:name_uniquer",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:dynamic_update_slice_util",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:ir_builder_mixin",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "//xla/service/llvm_ir:math_ops",
        "//xla/service/llvm_ir:tuple_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "target_machine_features",
    srcs = [
        "target_machine_features.cc",
    ],
    hdrs = ["target_machine_features.h"],
    deps = [
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "target_machine_features_fake",
    testonly = 1,
    hdrs = ["target_machine_features_fake.h"],
    deps = [
        ":target_machine_features",
    ],
)

cc_library(
    name = "ir_function",
    srcs = ["ir_function.cc"],
    hdrs = ["ir_function.h"],
    deps = [
        ":cpu_runtime",
        ":ir_emission_utils",
        ":shape_partition",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "parallel_loop_emitter",
    srcs = ["parallel_loop_emitter.cc"],
    hdrs = ["parallel_loop_emitter.h"],
    deps = [
        ":ir_emission_utils",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "tiled_dot_emitter",
    srcs = ["tiled_dot_emitter.cc"],
    hdrs = ["tiled_dot_emitter.h"],
    deps = [
        ":vector_support_library",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:kernel_support_library",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "dot_op_emitter",
    srcs = ["dot_op_emitter.cc"],
    hdrs = [
        "dot_op_emitter.h",
    ],
    deps = [
        ":backend_config_proto_cc",
        ":cpu_options",
        ":cpu_runtime",
        ":ir_emission_utils",
        ":mlir_emitter",
        ":target_machine_features",
        ":tiled_dot_emitter",
        ":vector_support_library",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:kernel_support_library",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "sample_harness_build_test",
    targets = [
        ":sample_harness",
    ],
)

xla_cc_binary(
    name = "sample_harness",
    srcs = ["sample_harness.cc"],
    deps = [
        "//xla:array4d",
        "//xla:literal",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "compiler_functor",
    srcs = ["compiler_functor.cc"],
    hdrs = ["compiler_functor.h"],
    deps = [
        ":cpu_runtime",
        ":llvm_ir_runtime",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/functional:any_invocable",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cpu_runtime",
    srcs = [
        "cpu_runtime.cc",
        "xfeed_manager.cc",
    ],
    hdrs = [
        "cpu_runtime.h",
        "xfeed_manager.h",
    ],
    copts = runtime_copts(),
    deps = [
        ":collectives_interface",
        ":cpu_executable_run_options",
        ":in_process_collectives",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/service:global_device_id",
        "//xla/service:hlo_parser",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "llvm_ir_runtime",
    srcs = [
        "llvm_ir_runtime.cc",
    ],
    hdrs = [
        "llvm_ir_runtime.h",
    ],
    deps = [
        ":vector_support_library",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:math_ops",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TransformUtils",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "runtime_conv2d",
    srcs = [
        "runtime_conv2d.cc",
        "runtime_conv_impl.h",
    ],
    hdrs = ["runtime_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
        "@local_tsl//tsl/platform:mutex",  # build_cleaner: keep
    ],
)

cc_library(
    name = "runtime_conv3d",
    srcs = [
        "runtime_conv3d.cc",
        "runtime_conv_impl.h",
    ],
    hdrs = ["runtime_conv3d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
        "@local_tsl//tsl/platform:mutex",  # build_cleaner: keep
    ],
)

cc_library(
    name = "runtime_custom_call_status",
    srcs = ["runtime_custom_call_status.cc"],
    hdrs = ["runtime_custom_call_status.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:custom_call_status_internal",
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "runtime_conv2d_mkl",
    srcs = [
        "runtime_conv2d_mkl.cc",
    ],
    hdrs = ["runtime_conv2d_mkl.h"],
    copts = runtime_copts() + tf_openmp_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_conv2d",
        ":runtime_single_threaded_conv2d",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
    ] + mkl_deps(),
)

cc_library(
    name = "runtime_fft",
    srcs = [
        "runtime_fft.cc",
    ],
    hdrs = ["runtime_fft.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@ducc//:fft_wrapper",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "runtime_matmul",
    srcs = [
        "runtime_matmul_c128.cc",
        "runtime_matmul_c64.cc",
        "runtime_matmul_common.h",
        "runtime_matmul_f16.cc",
        "runtime_matmul_f32.cc",
        "runtime_matmul_f64.cc",
        "runtime_matmul_s32.cc",
    ],
    hdrs = ["runtime_matmul.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel",
        "@local_tsl//tsl/platform:mutex",  # build_cleaner: keep
    ],
)

cc_library(
    name = "runtime_matmul_acl",
    srcs = ["runtime_matmul_acl.cc"],
    hdrs = ["runtime_matmul_acl.h"],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        ":runtime_matmul",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:dynamic_annotations",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:types",
    ] + acl_deps(),
)

cc_library(
    name = "runtime_conv2d_acl",
    srcs = [
        "runtime_conv2d_acl.cc",
    ],
    hdrs = ["runtime_conv2d_acl.h"],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_conv2d",
        ":runtime_lightweight_check",
        ":runtime_single_threaded_conv2d",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
        "@local_tsl//tsl/platform:dynamic_annotations",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:types",
    ] + acl_deps(),
)

cc_library(
    name = "runtime_single_threaded_conv2d",
    srcs = [
        "runtime_conv_impl.h",
        "runtime_single_threaded_conv2d.cc",
    ],
    hdrs = ["runtime_single_threaded_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
        "@local_tsl//tsl/platform:mutex",  # build_cleaner: keep
    ],
)

cc_library(
    name = "runtime_single_threaded_conv3d",
    srcs = [
        "runtime_conv_impl.h",
        "runtime_single_threaded_conv3d.cc",
    ],
    hdrs = ["runtime_single_threaded_conv3d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel",
        "@local_tsl//tsl/framework/convolution:eigen_helpers",
        "@local_tsl//tsl/platform:mutex",  # build_cleaner: keep
    ],
)

cc_library(
    name = "runtime_single_threaded_fft",
    srcs = [
        "runtime_single_threaded_fft.cc",
    ],
    hdrs = ["runtime_single_threaded_fft.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_fft",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "runtime_single_threaded_matmul_impl",
    srcs = [
        "runtime_single_threaded_matmul_c128.cc",
        "runtime_single_threaded_matmul_c64.cc",
        "runtime_single_threaded_matmul_common.h",
        "runtime_single_threaded_matmul_f16.cc",
        "runtime_single_threaded_matmul_f32.cc",
        "runtime_single_threaded_matmul_f64.cc",
        "runtime_single_threaded_matmul_s32.cc",
    ],
    hdrs = ["runtime_single_threaded_matmul.h"],
    compatible_with = get_compatible_with_portable(),
    copts = runtime_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel_no_mkl",
    ],
)

cc_library(
    name = "runtime_single_threaded_matmul",
    hdrs = ["runtime_single_threaded_matmul.h"],
    compatible_with = get_compatible_with_portable(),
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_single_threaded_matmul_impl",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "runtime_single_threaded_matmul_nomkl",
    compatible_with = get_compatible_with_portable(),
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_single_threaded_matmul_impl",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework/contraction:eigen_contraction_kernel_no_mkl",
    ],
)

cc_library(
    name = "runtime_key_value_sort",
    srcs = ["runtime_key_value_sort.cc"],
    hdrs = ["runtime_key_value_sort.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "runtime_topk",
    srcs = ["runtime_topk.cc"],
    hdrs = ["runtime_topk.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "runtime_fork_join",
    srcs = ["runtime_fork_join.cc"],
    hdrs = ["runtime_fork_join.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:executable_run_options",
        "//xla/service:custom_call_status_internal",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "runtime_handle_ffi_call",
    srcs = ["runtime_handle_ffi_call.cc"],
    hdrs = ["runtime_handle_ffi_call.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi:call_frame",
        "//xla/ffi:ffi_api",
        "//xla/service:custom_call_status_public_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "cpu_runtime_test",
    srcs = ["cpu_runtime_test.cc"],
    shard_count = 10,
    tags = ["optonly"],
    deps = [
        ":cpu_runtime",
        ":runtime_custom_call_status",
        ":runtime_matmul",
        ":runtime_matmul_acl",
        ":runtime_single_threaded_matmul",
        "//xla:array2d",
        "//xla:types",
        "//xla/client:local_client",
        "//xla/service:custom_call_status_internal",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "cpu_instruction_fusion_test",
    srcs = ["cpu_instruction_fusion_test.cc"],
    tags = ["no_aarch64"],
    deps = [
        ":cpu_instruction_fusion",
        "//xla:shape_util",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:transpose_folding",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "xfeed_manager_test",
    size = "small",
    srcs = ["xfeed_manager_test.cc"],
    deps = [
        ":cpu_runtime",
        "//xla:shape_util",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cpu_instruction_fusion",
    srcs = ["cpu_instruction_fusion.cc"],
    hdrs = ["cpu_instruction_fusion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:fusion_node_indexing_evaluation",
        "//xla/service:instruction_fusion",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    deps = [
        ":cpu_runtime",
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "@llvm-project//llvm:Core",
    ],
)

xla_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        ":target_machine_features_fake",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "cpu_layout_assignment",
    srcs = ["cpu_layout_assignment.cc"],
    hdrs = ["cpu_layout_assignment.h"],
    deps = [
        ":dot_op_emitter",
        ":ir_emission_utils",
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:layout_assignment",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "cpu_layout_assignment_test",
    size = "small",
    srcs = ["cpu_layout_assignment_test.cc"],
    deps = [
        ":cpu_layout_assignment",
        ":target_machine_features_fake",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:algebraic_simplifier",
        "//xla/service:computation_layout",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "conv_canonicalization",
    srcs = ["conv_canonicalization.cc"],
    hdrs = ["conv_canonicalization.h"],
    deps = [
        ":cpu_runtime",
        ":ir_emission_utils",
        ":target_machine_features",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "conv_canonicalization_test",
    srcs = ["conv_canonicalization_test.cc"],
    deps = [
        ":conv_canonicalization",
        ":target_machine_features_fake",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "shape_partition",
    srcs = ["shape_partition.cc"],
    hdrs = ["shape_partition.h"],
    deps = [
        "//xla:shape_util",
    ],
)

xla_cc_test(
    name = "shape_partition_test",
    srcs = ["shape_partition_test.cc"],
    deps = [
        ":shape_partition",
        "//xla:test_helpers",
        "//xla:util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "parallel_task_assignment",
    srcs = ["parallel_task_assignment.cc"],
    hdrs = ["parallel_task_assignment.h"],
    deps = [
        ":backend_config_proto_cc",
        ":ir_emission_utils",
        ":shape_partition",
        ":target_machine_features",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_pass",
        "//xla/service/llvm_ir:dynamic_update_slice_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "parallel_task_assignment_test",
    srcs = ["parallel_task_assignment_test.cc"],
    deps = [
        ":cpu_executable",
        ":parallel_task_assignment",
        ":target_machine_features_fake",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "cpu_options",
    srcs = ["cpu_options.cc"],
    hdrs = ["cpu_options.h"],
    deps = [
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "orc_jit_memory_mapper",
    srcs = ["orc_jit_memory_mapper.cc"],
    hdrs = ["orc_jit_memory_mapper.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:ExecutionEngine",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "vector_support_library",
    srcs = ["vector_support_library.cc"],
    hdrs = ["vector_support_library.h"],
    deps = [
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "cpu_eigen_tensor_alignment_test",
    size = "small",
    srcs = ["cpu_eigen_tensor_alignment_test.cc"],
    deps = [
        ":ir_emission_utils",
        ":target_machine_features_fake",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "vectorized_reduce_with_no_vector_registers_test",
    size = "small",
    srcs = ["vectorized_reduce_with_no_vector_registers_test.cc"],
    tags = ["no_aarch64"],
    deps = [
        ":cpu_compiler",
        ":cpu_transfer_manager",
        ":test_header_helper",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "mlir_emitter",
    srcs = ["mlir_emitter.cc"],
    hdrs = ["mlir_emitter.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
    ],
)

tf_proto_library(
    name = "backend_config_proto",
    srcs = ["backend_config.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "onednn_util",
    srcs = ["onednn_util.cc"],
    hdrs = [
        "onednn_util.h",
        "//xla/tsl/util:onednn_util_hdrs",
    ],
    copts = runtime_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_memory_util",
    srcs = ["onednn_memory_util.cc"],
    hdrs = ["onednn_memory_util.h"],
    copts = runtime_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:ir_builder_mixin",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_matmul",
    srcs = ["onednn_matmul.cc"],
    hdrs = ["onednn_matmul.h"],
    copts = runtime_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":backend_config_proto_cc",
        ":onednn_memory_util",
        ":onednn_util",
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_layer_norm",
    srcs = ["onednn_layer_norm.cc"],
    hdrs = [
        "onednn_layer_norm.h",
        "//xla/tsl/util:onednn_util_hdrs",
    ],
    copts = runtime_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":backend_config_proto_cc",
        ":onednn_memory_util",
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_softmax",
    srcs = ["onednn_softmax.cc"],
    hdrs = [
        "onednn_softmax.h",
        "//xla/tsl/util:onednn_util_hdrs",
    ],
    copts = runtime_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":backend_config_proto_cc",
        ":onednn_memory_util",
        ":runtime_lightweight_check",
        "//xla:executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/log:check",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_matmul_rewriter",
    srcs = ["onednn_matmul_rewriter.cc"],
    hdrs = [
        "onednn_matmul.h",
        "onednn_matmul_rewriter.h",
        "//xla/tsl/util:onednn_util_hdrs",
    ],
    copts = tsl_copts(),
    deps = [
        ":backend_config_proto_cc",
        ":onednn_matmul",
        ":onednn_memory_util",
        ":onednn_util",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "onednn_ops_rewriter",
    srcs = ["onednn_ops_rewriter.cc"],
    hdrs = ["onednn_ops_rewriter.h"],
    copts = tsl_copts(),
    deps = [
        ":backend_config_proto_cc",
        ":onednn_memory_util",
        ":onednn_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@local_tsl//tsl/platform:platform_port",
    ] + mkl_deps(),
)

cc_library(
    name = "cpu_float_support",
    srcs = ["cpu_float_support.cc"],
    hdrs = ["cpu_float_support.h"],
    copts = tsl_copts(),
    deps = [
        ":onednn_matmul_rewriter",
        "//xla/service:float_support",
    ],
)

cc_library(
    name = "cpu_symbol_repository",
    hdrs = ["cpu_symbol_repository.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/service:symbol_repository",
    ],
)

cc_library(
    name = "collectives_interface",
    hdrs = ["collectives_interface.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "in_process_collectives",
    srcs = ["in_process_collectives.cc"],
    hdrs = ["in_process_collectives.h"],
    deps = [
        ":collectives_interface",
        "//xla:refcounting_hash_map",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "cpu_executable_run_options",
    hdrs = ["cpu_executable_run_options.h"],
    deps = [":collectives_interface"],
)
