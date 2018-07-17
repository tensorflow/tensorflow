# Description:
#   GPU-specific components in XLA service implementation.

load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

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

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

xla_proto_library(
    name = "backend_configs",
    srcs = ["backend_configs.proto"],
)

cc_library(
    name = "gpu_constants",
    srcs = ["gpu_constants.cc"],
    hdrs = ["gpu_constants.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "partition_assignment",
    srcs = [
        "partition_assignment.cc",
    ],
    hdrs = [
        "partition_assignment.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

# TODO(b/29140563) This target is flaky, disabled until flakiness is
# root-caused. Failed on 2016-06-08.
#tf_cc_test(
#    name = "partition_assignment_test",
#    srcs = [
#        "partition_assignment_test.cc",
#    ],
#    tags = [
#        "requires-gpu-sm35",
#    ],
#    deps = [
#        ":partition_assignment",
#        "//tensorflow/core:stream_executor_no_cuda",
#        "//tensorflow/compiler/xla:shape_util",
#        "//tensorflow/compiler/xla:xla_data_proto",
#        "//tensorflow/compiler/xla/service:gpu_plugin",
#        "//tensorflow/compiler/xla/service:hlo",
#        "//tensorflow/compiler/xla/tests:hlo_test_base",
#        "//tensorflow/core:test_main",
#    ],
#)

cc_library(
    name = "stream_assignment",
    srcs = ["stream_assignment.cc"],
    hdrs = ["stream_assignment.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "stream_assignment_test",
    srcs = [
        "stream_assignment_test.cc",
    ],
    deps = [
        ":stream_assignment",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_to_ir_bindings",
    srcs = ["hlo_to_ir_bindings.cc"],
    hdrs = ["hlo_to_ir_bindings.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "ir_emitter",
    srcs = [
        "ir_emitter.cc",
        "ir_emitter_nested.cc",
        "ir_emitter_unnested.cc",
    ],
    hdrs = [
        "ir_emitter.h",
        "ir_emitter_context.h",
        "ir_emitter_nested.h",
        "ir_emitter_unnested.h",
    ],
    deps = [
        ":backend_configs",
        ":cudnn_convolution_runner",
        ":elemental_ir_emitter",
        ":gpu_constants",
        ":gpu_executable",
        ":hlo_to_ir_bindings",
        ":ir_emission_utils",
        ":parallel_loop_emitter",
        ":partition_assignment",
        ":while_transformer",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_tiling",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ops",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:core",
        "@llvm//:support",
    ],
)

cc_library(
    name = "parallel_loop_emitter",
    srcs = ["parallel_loop_emitter.cc"],
    hdrs = ["parallel_loop_emitter.h"],
    deps = [
        ":partition_assignment",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
        "@llvm//:core",
        "@llvm//:support",
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
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "hlo_execution_profiler",
    srcs = ["hlo_execution_profiler.cc"],
    hdrs = ["hlo_execution_profiler.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:pool",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "gpu_executable",
    srcs = [
        "conditional_thunk.cc",
        "convolution_thunk.cc",
        "copy_thunk.cc",
        "cudnn_batchnorm_thunk.cc",
        "fft_thunk.cc",
        "for_thunk.cc",
        "gemm_thunk.cc",
        "gpu_executable.cc",
        "infeed_thunk.cc",
        "kernel_thunk.cc",
        "memset_thunk.cc",
        "outfeed_thunk.cc",
        "sequential_thunk.cc",
        "thunk_schedule.cc",
        "tuple_thunk.cc",
        "while_thunk.cc",
    ],
    hdrs = [
        "conditional_thunk.h",
        "convolution_thunk.h",
        "copy_thunk.h",
        "cudnn_batchnorm_thunk.h",
        "fft_thunk.h",
        "for_thunk.h",
        "gemm_thunk.h",
        "gpu_executable.h",
        "infeed_thunk.h",
        "kernel_thunk.h",
        "memset_thunk.h",
        "outfeed_thunk.h",
        "sequential_thunk.h",
        "thunk.h",
        "thunk_schedule.h",
        "tuple_thunk.h",
        "while_thunk.h",
    ],
    deps = [
        ":buffer_allocations",
        ":cudnn_convolution_runner",
        ":hlo_execution_profiler",
        ":infeed_manager",
        ":ir_emission_utils",
        ":outfeed_manager",
        ":partition_assignment",
        ":stream_assignment",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:tuple_points_to_analysis",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
        "//tensorflow/core/platform/default/build_config:cudnn_plugin",
        "//tensorflow/core/platform/default/build_config:cufft_plugin",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",  # build_cleaner: keep
        "//tensorflow/stream_executor",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "cudnn_convolution_algorithm_picker",
    srcs = ["cudnn_convolution_algorithm_picker.cc"],
    hdrs = ["cudnn_convolution_algorithm_picker.h"],
    deps = [
        ":backend_configs",
        ":cudnn_convolution_runner",
        ":gpu_executable",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "cudnn_convolution_runner",
    srcs = ["cudnn_convolution_runner.cc"],
    hdrs = ["cudnn_convolution_runner.h"],
    deps = [
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "cudnn_convolution_rewriter",
    srcs = ["cudnn_convolution_rewriter.cc"],
    hdrs = ["cudnn_convolution_rewriter.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "cudnn_convolution_rewriter_test",
    srcs = ["cudnn_convolution_rewriter_test.cc"],
    deps = [
        ":cudnn_convolution_rewriter",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "//tensorflow/compiler/xla/service:pattern_matcher",
    ],
)

tf_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":instruction_fusion",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:multi_output_fusion",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "multi_output_fusion_test",
    srcs = ["multi_output_fusion_test.cc"],
    deps = [
        ":multi_output_fusion",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_copy_insertion",
    srcs = ["gpu_copy_insertion.cc"],
    hdrs = ["gpu_copy_insertion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla/service:call_graph",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "fusion_merger",
    srcs = ["fusion_merger.cc"],
    hdrs = ["fusion_merger.h"],
    deps = [
        ":instruction_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "fusion_merger_test",
    srcs = ["fusion_merger_test.cc"],
    deps = [
        ":fusion_merger",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "pad_insertion",
    srcs = ["pad_insertion.cc"],
    hdrs = ["pad_insertion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shape_inference",
    ],
)

cc_library(
    name = "gpu_transfer_manager",
    srcs = ["gpu_transfer_manager.cc"],
    hdrs = ["gpu_transfer_manager.h"],
    deps = [
        ":gpu_compiler",
        ":outfeed_manager",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service/gpu:infeed_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:core",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_compiler",
    srcs = ["gpu_compiler.cc"],
    hdrs = ["gpu_compiler.h"],
    deps = [
        ":cudnn_convolution_algorithm_picker",
        ":cudnn_convolution_rewriter",
        ":fusion_merger",
        ":gpu_constants",
        ":gpu_copy_insertion",
        ":gpu_executable",
        ":gpu_hlo_support_checker",
        ":gpu_layout_assignment",
        ":hlo_schedule",
        ":instruction_fusion",
        ":ir_emission_utils",
        ":ir_emitter",
        ":multi_output_fusion",
        ":pad_insertion",
        ":partition_assignment",
        ":stream_assignment",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:batchnorm_expander",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:buffer_liveness",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:conditional_simplifier",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_element_type_converter",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:reduce_precision_insertion",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:while_loop_invariant_code_motion",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/gpu:cudnn_batchnorm_rewriter",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:core",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "cudnn_batchnorm_rewriter",
    srcs = ["cudnn_batchnorm_rewriter.cc"],
    hdrs = ["cudnn_batchnorm_rewriter.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

cc_library(
    name = "xfeed_queue",
    hdrs = ["xfeed_queue.h"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "infeed_manager",
    srcs = ["infeed_manager.cc"],
    hdrs = ["infeed_manager.h"],
    deps = [
        ":xfeed_queue",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "outfeed_manager",
    srcs = ["outfeed_manager.cc"],
    hdrs = ["outfeed_manager.h"],
    deps = [
        ":xfeed_queue",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_layout_assignment",
    srcs = ["gpu_layout_assignment.cc"],
    hdrs = ["gpu_layout_assignment.h"],
    deps = [
        ":gpu_options",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

tf_cc_test(
    name = "gpu_layout_assignment_test",
    srcs = ["gpu_layout_assignment_test.cc"],
    deps = [
        ":gpu_layout_assignment",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "hlo_schedule",
    srcs = ["hlo_schedule.cc"],
    hdrs = ["hlo_schedule.h"],
    deps = [
        ":stream_assignment",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_value",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:hlo_scheduling",
    ],
)

tf_cc_test(
    name = "hlo_schedule_test",
    srcs = [
        "hlo_schedule_test.cc",
    ],
    deps = [
        ":hlo_schedule",
        ":stream_assignment",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "while_transformer",
    srcs = ["while_transformer.cc"],
    hdrs = ["while_transformer.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "while_transformer_test",
    srcs = ["while_transformer_test.cc"],
    deps = [
        ":instruction_fusion",
        ":while_transformer",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "gpu_hlo_support_checker",
    srcs = ["gpu_hlo_support_checker.cc"],
    hdrs = ["gpu_hlo_support_checker.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_options",
    srcs = ["gpu_options.cc"],
    hdrs = ["gpu_options.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "stream_executor_util",
    srcs = ["stream_executor_util.cc"],
    hdrs = ["stream_executor_util.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

tf_cc_test(
    name = "gpu_hlo_support_checker_test",
    srcs = ["gpu_hlo_support_checker_test.cc"],
    deps = [
        ":gpu_hlo_support_checker",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
    ],
)
