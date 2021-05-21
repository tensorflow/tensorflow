# Description:
#    LLVM-based CPU backend for XLA.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "ORC_JIT_MEMORY_MAPPER_TARGETS")
load(
    "//third_party/mkl:build_defs.bzl",
    "mkl_deps",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_openmp_copts")
load(":build_defs.bzl", "runtime_copts")
load("//tensorflow/core/platform:build_config.bzl", "if_llvm_system_z_available")

package(
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

cc_library(
    name = "test_header_helper",
    testonly = True,
    hdrs = ["test_target_triple_helper.h"],
    deps = [
        "//tensorflow/core:test",
    ],
)

filegroup(
    name = "runtime_srcs",
    srcs = [
        # Single-threaded support.
        "runtime_fp16.cc",
        "runtime_key_value_sort.cc",
        "runtime_pow.cc",
        "runtime_single_threaded_conv2d.cc",
        "runtime_single_threaded_fft.cc",
        "runtime_single_threaded_matmul.cc",
        "runtime_topk.cc",
    ] + [
        # Multi-threaded support.
        "runtime_conv2d.cc",
        "runtime_fft.cc",
        "runtime_matmul.cc",
        "runtime_fork_join.cc",
    ],
    visibility = [":friends"],
)

filegroup(
    name = "runtime_hdrs",
    srcs = [
        # Single-threaded support.
        "runtime_conv2d_impl.h",
        "runtime_fft_impl.h",
        "runtime_fp16.h",
        "runtime_key_value_sort.h",
        "runtime_pow.h",
        "runtime_single_threaded_conv2d.h",
        "runtime_single_threaded_fft.h",
        "runtime_single_threaded_matmul.h",
        "runtime_topk.h",
    ] + [
        # Multi-threaded support.
        "runtime_conv2d.h",
        "runtime_fft.h",
        "runtime_fork_join.h",
        "runtime_lightweight_check.h",
        "runtime_matmul.h",
    ],
    visibility = [":friends"],
)

cc_library(
    name = "cpu_xfeed",
    srcs = ["cpu_xfeed.cc"],
    hdrs = ["cpu_xfeed.h"],
    deps = [
        ":cpu_runtime",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cpu_transfer_manager",
    srcs = ["cpu_transfer_manager.cc"],
    hdrs = ["cpu_transfer_manager.h"],
    deps = [
        ":cpu_runtime",
        ":cpu_xfeed",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "buffer_info_util",
    srcs = ["buffer_info_util.cc"],
    hdrs = ["buffer_info_util.h"],
    deps = [
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cpu_compiler",
    srcs = ["cpu_compiler.cc"],
    hdrs = ["cpu_compiler.h"],
    deps = [
        ":compiler_functor",
        ":buffer_info_util",
        ":conv_canonicalization",
        ":cpu_executable",
        ":cpu_instruction_fusion",
        ":cpu_layout_assignment",
        ":cpu_options",
        ":dot_op_emitter",
        ":ir_emission_utils",
        ":ir_emitter",
        ":parallel_task_assignment",
        ":simple_orc_jit",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        ":target_machine_features",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:VectorOps",
        "//tensorflow/compiler/xla/service:all_gather_decomposer",
        "//tensorflow/compiler/xla/service:all_to_all_decomposer",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:result_caster",
        "//tensorflow/compiler/xla/service:topk_rewriter",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:rng_bit_generator_expander",
        "//tensorflow/compiler/xla/service:tree_reduction_rewriter",
        "//tensorflow/compiler/xla/service:conditional_canonicalizer",
        "//tensorflow/compiler/xla/service:conditional_to_select",
        "//tensorflow/compiler/xla/service:slow_operation_alarm",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "//tensorflow/compiler/xla/service:comparison_expander",
        "//tensorflow/compiler/xla/service:slice_sinker",
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla/service:operand_upcaster",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:logistic_expander",
        "//tensorflow/compiler/xla/service:batch_dot_simplification",
        "//tensorflow/compiler/xla/service:batchnorm_expander",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:cholesky_expander",
        "//tensorflow/compiler/xla/service:eigh_expander",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:conditional_simplifier",
        "//tensorflow/compiler/xla/service:convolution_group_converter",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:dynamic_padder",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_element_type_converter",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:indexed_array_analysis",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:rng_expander",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/service:triangular_solve_expander",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:while_loop_invariant_code_motion",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ] + select({
        "//tensorflow:arm_any": [
            "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
        ],
        "//tensorflow:linux_ppc64le": [
            "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
        ],
        "//tensorflow:macos_arm64": [
            "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
        ],
        "//conditions:default": [
        ],
    }) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "simple_orc_jit",
    srcs = [
        "simple_orc_jit.cc",
        "windows_compatibility.cc",
        "windows_compatibility.h",
    ],
    hdrs = ["simple_orc_jit.h"],
    deps = [
        ":compiler_functor",
        ":cpu_runtime",
        ":orc_jit_memory_mapper",
        ":runtime_fp16",
        ":runtime_pow",
        ":runtime_conv2d",
        ":runtime_conv2d_mkl",
        ":runtime_fft",
        ":runtime_fork_join",
        ":runtime_key_value_sort",
        ":runtime_topk",
        ":runtime_matmul",
        ":runtime_matmul_mkl",
        ":runtime_single_threaded_conv2d",
        ":runtime_single_threaded_fft",
        ":runtime_single_threaded_matmul",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",  # fixdeps: keep
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",  # fixdeps: keep
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ] + ORC_JIT_MEMORY_MAPPER_TARGETS,
)

cc_library(
    name = "runtime_lightweight_check",
    hdrs = ["runtime_lightweight_check.h"],
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
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
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
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "cpu_executable",
    srcs = ["cpu_executable.cc"],
    hdrs = ["cpu_executable.h"],
    deps = [
        ":simple_orc_jit",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor/host:host_stream",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:OrcJIT",
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
    deps = [
        ":cpu_options",
        ":cpu_runtime",
        ":dot_op_emitter",
        ":ir_emission_utils",
        ":ir_function",
        ":parallel_loop_emitter",
        ":shape_partition",
        ":simple_orc_jit",
        ":target_machine_features",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:dynamic_update_slice_util",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_builder_mixin",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "target_machine_features",
    srcs = [
        "target_machine_features.cc",
    ],
    hdrs = ["target_machine_features.h"],
    deps = [
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Target",
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
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
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
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
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
        ":cpu_options",
        ":cpu_runtime",
        ":ir_emission_utils",
        ":mlir_emitter",
        ":target_machine_features",
        ":tiled_dot_emitter",
        ":vector_support_library",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:StandardOps",
    ],
)

tf_cc_binary(
    name = "sample_harness",
    srcs = ["sample_harness.cc"],
    deps = [
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "compiler_functor",
    srcs = ["compiler_functor.cc"],
    hdrs = ["compiler_functor.h"],
    deps = [
        ":cpu_runtime",
        ":llvm_ir_runtime",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
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
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:refcounting_hash_map",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:math_ops",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TransformUtils",
    ],
)

cc_library(
    name = "runtime_conv2d",
    srcs = [
        "runtime_conv2d.cc",
        "runtime_conv2d_impl.h",
    ],
    hdrs = ["runtime_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/kernels:eigen_contraction_kernel",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
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
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ] + mkl_deps(),
)

cc_library(
    name = "runtime_fft",
    srcs = [
        "runtime_fft.cc",
        "runtime_fft_impl.h",
    ],
    hdrs = ["runtime_fft.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_matmul",
    srcs = ["runtime_matmul.cc"],
    hdrs = ["runtime_matmul.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/kernels:eigen_contraction_kernel",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_matmul_mkl",
    srcs = ["runtime_matmul_mkl.cc"],
    hdrs = ["runtime_matmul_mkl.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ] + mkl_deps(),
)

cc_library(
    name = "runtime_single_threaded_conv2d",
    srcs = [
        "runtime_conv2d_impl.h",
        "runtime_single_threaded_conv2d.cc",
    ],
    hdrs = ["runtime_single_threaded_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_lightweight_check",
        "//tensorflow/core/kernels:eigen_contraction_kernel",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_single_threaded_fft",
    srcs = [
        "runtime_fft_impl.h",
        "runtime_single_threaded_fft.cc",
    ],
    hdrs = ["runtime_single_threaded_fft.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_single_threaded_matmul",
    srcs = ["runtime_single_threaded_matmul.cc"],
    hdrs = ["runtime_single_threaded_matmul.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:eigen_contraction_kernel",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_key_value_sort",
    srcs = ["runtime_key_value_sort.cc"],
    hdrs = ["runtime_key_value_sort.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_topk",
    srcs = ["runtime_topk.cc"],
    hdrs = ["runtime_topk.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "runtime_fork_join",
    srcs = ["runtime_fork_join.cc"],
    hdrs = ["runtime_fork_join.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "cpu_runtime_test",
    srcs = ["cpu_runtime_test.cc"],
    shard_count = 10,
    tags = ["optonly"],
    deps = [
        ":cpu_runtime",
        ":runtime_matmul",
        ":runtime_matmul_mkl",
        ":runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "runtime_fft_test",
    srcs = [
        "runtime_fft_impl.h",
        "runtime_fft_test.cc",
    ],
    deps = [
        ":runtime_single_threaded_fft",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:numeric_types",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "cpu_instruction_fusion_test",
    srcs = ["cpu_instruction_fusion_test.cc"],
    deps = [
        ":cpu_instruction_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "xfeed_manager_test",
    size = "small",
    srcs = ["xfeed_manager_test.cc"],
    deps = [
        ":cpu_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "cpu_instruction_fusion",
    srcs = ["cpu_instruction_fusion.cc"],
    hdrs = ["cpu_instruction_fusion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla/service:fusion_node_indexing_evaluation",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
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
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
        "@llvm-project//llvm:Core",
    ],
)

tf_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        ":target_machine_features_fake",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
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
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "cpu_layout_assignment_test",
    size = "small",
    srcs = ["cpu_layout_assignment_test.cc"],
    deps = [
        ":cpu_layout_assignment",
        ":target_machine_features_fake",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "conv_canonicalization_test",
    srcs = ["conv_canonicalization_test.cc"],
    deps = [
        ":conv_canonicalization",
        ":target_machine_features_fake",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "shape_partition",
    srcs = ["shape_partition.cc"],
    hdrs = ["shape_partition.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
    ],
)

tf_cc_test(
    name = "shape_partition_test",
    srcs = ["shape_partition_test.cc"],
    deps = [
        ":shape_partition",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "parallel_task_assignment",
    srcs = ["parallel_task_assignment.cc"],
    hdrs = ["parallel_task_assignment.h"],
    deps = [
        ":dot_op_emitter",
        ":ir_emission_utils",
        ":shape_partition",
        ":target_machine_features",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service/llvm_ir:dynamic_update_slice_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "parallel_task_assignment_test",
    srcs = ["parallel_task_assignment_test.cc"],
    deps = [
        ":cpu_executable",
        ":parallel_task_assignment",
        ":target_machine_features_fake",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "cpu_options",
    srcs = ["cpu_options.cc"],
    hdrs = ["cpu_options.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "orc_jit_memory_mapper",
    srcs = ["orc_jit_memory_mapper.cc"],
    hdrs = ["orc_jit_memory_mapper.h"],
    deps = [
        "//tensorflow/core:lib",
        "@llvm-project//llvm:ExecutionEngine",
    ],
)

cc_library(
    name = "vector_support_library",
    srcs = ["vector_support_library.cc"],
    hdrs = ["vector_support_library.h"],
    deps = [
        ":target_machine_features",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "cpu_eigen_tensor_alignment_test",
    size = "small",
    srcs = ["cpu_eigen_tensor_alignment_test.cc"],
    deps = [
        ":ir_emission_utils",
        ":target_machine_features_fake",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "vectorized_reduce_with_no_vector_registers_test",
    size = "small",
    srcs = ["vectorized_reduce_with_no_vector_registers_test.cc"],
    deps = [
        ":cpu_compiler",
        ":cpu_transfer_manager",
        ":test_header_helper",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "mlir_emitter",
    srcs = ["mlir_emitter.cc"],
    hdrs = ["mlir_emitter.h"],
    deps = [
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
    ],
)
