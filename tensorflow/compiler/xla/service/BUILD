# Description:
#   XLA service implementation.

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

xla_proto_library(
    name = "session_proto",
    srcs = ["session.proto"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

xla_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    deps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

xla_proto_library(
    name = "hlo_profile_printer_data",
    srcs = ["hlo_profile_printer_data.proto"],
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
    name = "bfloat16_support",
    srcs = ["bfloat16_support.cc"],
    hdrs = ["bfloat16_support.h"],
    deps = [
        ":hlo",
    ],
)

cc_library(
    name = "bfloat16_conversion_folding",
    srcs = ["bfloat16_conversion_folding.cc"],
    hdrs = ["bfloat16_conversion_folding.h"],
    deps = [
        ":bfloat16_support",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "bfloat16_conversion_folding_test",
    srcs = ["bfloat16_conversion_folding_test.cc"],
    deps = [
        ":bfloat16_conversion_folding",
        ":bfloat16_support",
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "bfloat16_normalization",
    srcs = ["bfloat16_normalization.cc"],
    hdrs = ["bfloat16_normalization.h"],
    deps = [
        ":bfloat16_support",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "bfloat16_normalization_test",
    srcs = ["bfloat16_normalization_test.cc"],
    deps = [
        ":bfloat16_normalization",
        ":bfloat16_support",
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "bfloat16_propagation",
    srcs = ["bfloat16_propagation.cc"],
    hdrs = ["bfloat16_propagation.h"],
    deps = [
        ":bfloat16_support",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "bfloat16_propagation_test",
    srcs = ["bfloat16_propagation_test.cc"],
    deps = [
        ":bfloat16_propagation",
        ":bfloat16_support",
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_opcode_test",
    srcs = ["hlo_opcode_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_evaluator",
    srcs = ["hlo_evaluator.cc"],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        ":hlo",
        ":hlo_query",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/service:hlo_element_type_converter",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo",
    srcs = [
        "dfs_hlo_visitor.cc",
        "hlo_computation.cc",
        "hlo_instruction.cc",
        "hlo_module.cc",
        "hlo_opcode.cc",
        "hlo_sharding.cc",
    ],
    hdrs = [
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "hlo_computation.h",
        "hlo_instruction.h",
        "hlo_module.h",
        "hlo_opcode.h",
        "hlo_sharding.h",
    ],
    deps = [
        ":hlo_module_config",
        ":hlo_proto",
        ":hlo_reachability",
        ":name_uniquer",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "hlo_reachability",
    srcs = ["hlo_reachability.cc"],
    hdrs = ["hlo_reachability.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "hlo_reachability_test",
    srcs = ["hlo_reachability_test.cc"],
    deps = [
        ":hlo",
        ":hlo_reachability",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "hlo_matchers_test",
    srcs = ["hlo_matchers_test.cc"],
    deps = [
        ":hlo_matchers",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "versioned_computation_handle",
    srcs = ["versioned_computation_handle.cc"],
    hdrs = ["versioned_computation_handle.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_instruction_test",
    srcs = ["hlo_instruction_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "hlo_sharding_test",
    srcs = ["hlo_sharding_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "call_graph",
    srcs = ["call_graph.cc"],
    hdrs = ["call_graph.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":call_graph",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "flatten_call_graph",
    srcs = ["flatten_call_graph.cc"],
    hdrs = ["flatten_call_graph.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "call_inliner",
    srcs = ["call_inliner.cc"],
    hdrs = ["call_inliner.h"],
    deps = [
        ":call_graph",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "call_inliner_test",
    size = "small",
    srcs = ["call_inliner_test.cc"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_matchers",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "flatten_call_graph_test",
    srcs = ["flatten_call_graph_test.cc"],
    deps = [
        ":call_graph",
        ":flatten_call_graph",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "user_computation",
    srcs = ["user_computation.cc"],
    hdrs = ["user_computation.h"],
    deps = [
        ":hlo",
        ":session_proto",
        ":shape_inference",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "user_computation_test",
    srcs = ["user_computation_test.cc"],
    deps = [
        ":hlo_matchers",
        ":user_computation",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "platform_util",
    srcs = ["platform_util.cc"],
    hdrs = ["platform_util.h"],
    deps = [
        ":compiler",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":compiler",
        ":computation_placer",
        ":device_memory_allocator",
        ":platform_util",
        ":pool",
        ":transfer_manager",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":allocation_tracker",
        ":backend",
        ":channel_tracker",
        ":compilation_cache",
        ":compiler",
        ":computation_layout",
        ":computation_tracker",
        ":device_memory_allocator",
        ":executable",
        ":execution_tracker",
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_evaluator",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_proto_util",
        ":platform_util",
        ":session_proto",
        ":source_map_util",
        ":transfer_manager",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = 1,
)

cc_library(
    name = "local_service",
    srcs = ["local_service.cc"],
    hdrs = ["local_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":computation_tracker",
        ":device_memory_allocator",
        ":executable",
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":platform_util",
        ":service",
        ":shaped_buffer",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "compile_only_service",
    srcs = ["compile_only_service.cc"],
    hdrs = ["compile_only_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":computation_tracker",
        ":platform_util",
        ":service",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "cpu_plugin",
    deps = [
        ":service",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_transfer_manager",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "gpu_plugin",
    deps = [
        ":service",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_transfer_manager",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",
    ],
)

cc_library(
    name = "interpreter_plugin",
    deps = [
        ":service",
        "//tensorflow/compiler/xla/service/interpreter:compiler",
        "//tensorflow/compiler/xla/service/interpreter:interpreter_transfer_manager",
        "//tensorflow/compiler/xla/service/interpreter:platform",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "shaped_buffer",
    srcs = ["shaped_buffer.cc"],
    hdrs = ["shaped_buffer.h"],
    deps = [
        ":device_memory_allocator",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = [
        "executable.h",
        "service_executable_run_options.h",
    ],
    deps = [
        ":computation_layout",
        ":device_memory_allocator",
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        ":hlo_graph_dumper",
        ":pool",
        ":session_proto",
        ":shaped_buffer",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/stream_executor",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":executable",
        ":hlo",
        ":hlo_module_config",
        ":logical_buffer",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        ":compiler",
        "//tensorflow/core:lib_internal",
        "@llvm//:core",
    ],
)

cc_library(
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
        ":shaped_buffer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "allocation_tracker",
    srcs = ["allocation_tracker.cc"],
    hdrs = ["allocation_tracker.h"],
    deps = [
        ":backend",
        ":device_memory_allocator",
        ":transfer_manager",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "execution_tracker",
    srcs = ["execution_tracker.cc"],
    hdrs = ["execution_tracker.h"],
    deps = [
        ":backend",
        ":pool",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "computation_tracker",
    srcs = ["computation_tracker.cc"],
    hdrs = ["computation_tracker.h"],
    deps = [
        ":hlo",
        ":hlo_module_config",
        ":session_proto",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "channel_tracker",
    srcs = ["channel_tracker.cc"],
    hdrs = ["channel_tracker.h"],
    deps = [
        ":hlo",
        ":session_proto",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "name_uniquer_test",
    srcs = ["name_uniquer_test.cc"],
    deps = [
        ":name_uniquer",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "liveness_util",
    srcs = ["liveness_util.cc"],
    hdrs = ["liveness_util.h"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
    ],
)

tf_cc_test(
    name = "liveness_util_test",
    srcs = ["liveness_util_test.cc"],
    deps = [
        ":hlo",
        ":liveness_util",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_liveness",
    srcs = [
        "buffer_liveness.cc",
    ],
    hdrs = [
        "buffer_liveness.h",
    ],
    deps = [
        ":hlo",
        ":hlo_ordering",
        ":liveness_util",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "buffer_liveness_test",
    srcs = ["buffer_liveness_test.cc"],
    deps = [
        ":buffer_liveness",
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_assignment",
    srcs = [
        "buffer_assignment.cc",
    ],
    hdrs = [
        "buffer_assignment.h",
    ],
    deps = [
        ":buffer_liveness",
        ":heap_simulator",
        ":hlo",
        ":hlo_proto",
        ":hlo_scheduling",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "buffer_assignment_test",
    srcs = ["buffer_assignment_test.cc"],
    deps = [
        ":buffer_assignment",
        ":call_graph",
        ":computation_tracker",
        ":copy_insertion",
        ":cpu_plugin",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_ordering",
        ":hlo_scheduling",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_ordering",
    srcs = ["hlo_ordering.cc"],
    hdrs = ["hlo_ordering.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_proto",
        ":hlo_value",
        ":liveness_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_ordering_test",
    size = "small",
    srcs = ["hlo_ordering_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_scheduling",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/compiler/xla/tools/parser:hlo_parser",
    ],
)

cc_library(
    name = "heap_simulator",
    srcs = ["heap_simulator.cc"],
    hdrs = ["heap_simulator.h"],
    deps = [
        ":hlo",
        ":hlo_ordering",
        ":hlo_proto",
        ":liveness_util",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "heap_simulator_test",
    srcs = ["heap_simulator_test.cc"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_ordering",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_scheduling",
    srcs = ["hlo_scheduling.cc"],
    hdrs = ["hlo_scheduling.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_ordering",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_scheduling_test",
    srcs = ["hlo_scheduling_test.cc"],
    deps = [
        ":hlo",
        ":hlo_ordering",
        ":hlo_scheduling",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "batchnorm_expander",
    srcs = ["batchnorm_expander.cc"],
    hdrs = ["batchnorm_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "batchnorm_expander_test",
    size = "small",
    srcs = ["batchnorm_expander_test.cc"],
    deps = [
        ":batchnorm_expander",
        ":hlo",
        ":hlo_matchers",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "algebraic_simplifier",
    srcs = ["algebraic_simplifier.cc"],
    hdrs = ["algebraic_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "algebraic_simplifier_test",
    srcs = ["algebraic_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo",
        ":hlo_matchers",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "while_loop_simplifier",
    srcs = ["while_loop_simplifier.cc"],
    hdrs = ["while_loop_simplifier.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "while_loop_simplifier_test",
    srcs = ["while_loop_simplifier_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_loop_simplifier",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "defuser",
    srcs = ["defuser.cc"],
    hdrs = ["defuser.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "defuser_test",
    srcs = ["defuser_test.cc"],
    deps = [
        ":defuser",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
    ],
)

cc_library(
    name = "implicit_broadcast_remover",
    srcs = ["implicit_broadcast_remover.cc"],
    hdrs = ["implicit_broadcast_remover.h"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "implicit_broadcast_remover_test",
    srcs = ["implicit_broadcast_remover_test.cc"],
    deps = [
        ":hlo_matchers",
        ":implicit_broadcast_remover",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["dot_decomposer.cc"],
    hdrs = ["dot_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tuple_simplifier",
    srcs = ["tuple_simplifier.cc"],
    hdrs = ["tuple_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tuple_simplifier_test",
    srcs = ["tuple_simplifier_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "reshape_mover",
    srcs = ["reshape_mover.cc"],
    hdrs = ["reshape_mover.h"],
    deps = [
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "reshape_mover_test",
    srcs = ["reshape_mover_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":reshape_mover",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "inliner",
    srcs = ["inliner.cc"],
    hdrs = ["inliner.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "inliner_test",
    srcs = ["inliner_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo",
        ":hlo_matchers",
        ":inliner",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "computation_placer",
    srcs = ["computation_placer.cc"],
    hdrs = ["computation_placer.h"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = True,  # Contains per-platform computation placer registration
)

cc_library(
    name = "human_readable_profile_builder",
    srcs = ["human_readable_profile_builder.cc"],
    hdrs = ["human_readable_profile_builder.h"],
    deps = [
        "//tensorflow/compiler/xla:metric_table_report",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "generic_transfer_manager",
    srcs = ["generic_transfer_manager.cc"],
    hdrs = ["generic_transfer_manager.h"],
    deps = [
        ":transfer_manager",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/interpreter:platform_id",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "hlo_cost_analysis",
    srcs = ["hlo_cost_analysis.cc"],
    hdrs = ["hlo_cost_analysis.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "hlo_cost_analysis_test",
    srcs = ["hlo_cost_analysis_test.cc"],
    deps = [
        ":computation_tracker",
        ":cpu_plugin",
        ":hlo",
        ":hlo_cost_analysis",
        ":local_service",
        ":service",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_execution_profile",
    srcs = ["hlo_execution_profile.cc"],
    hdrs = ["hlo_execution_profile.h"],
    deps = [
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_profile_printer",
        ":human_readable_profile_builder",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

tf_cc_test(
    name = "hlo_execution_profile_test",
    srcs = ["hlo_execution_profile_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "hlo_computation_test",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_binary(
    name = "graphviz_example",
    srcs = ["graphviz_example.cc"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_module_test",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = ["logical_buffer.cc"],
    hdrs = ["logical_buffer.h"],
    deps = [
        ":hlo",
        ":hlo_proto",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "hlo_value",
    srcs = ["hlo_value.cc"],
    hdrs = ["hlo_value.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_dataflow_analysis",
    srcs = ["hlo_dataflow_analysis.cc"],
    hdrs = ["hlo_dataflow_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_dataflow_analysis_test",
    srcs = ["hlo_dataflow_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_buffer",
    srcs = ["hlo_buffer.cc"],
    hdrs = ["hlo_buffer.h"],
    deps = [
        ":hlo",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_alias_analysis",
    srcs = ["hlo_alias_analysis.cc"],
    hdrs = ["hlo_alias_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_alias_analysis_test",
    srcs = ["hlo_alias_analysis_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "logical_buffer_analysis",
    srcs = ["logical_buffer_analysis.cc"],
    hdrs = ["logical_buffer_analysis.h"],
    deps = [
        ":hlo",
        ":logical_buffer",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "tuple_points_to_analysis",
    srcs = ["tuple_points_to_analysis.cc"],
    hdrs = ["tuple_points_to_analysis.h"],
    deps = [
        ":hlo",
        ":logical_buffer",
        ":logical_buffer_analysis",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tuple_points_to_analysis_test",
    srcs = ["tuple_points_to_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":instruction_fusion",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":executable",
        ":hlo_module_config",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "layout_assignment",
    srcs = [
        "layout_assignment.cc",
    ],
    hdrs = [
        "layout_assignment.h",
    ],
    deps = [
        ":computation_layout",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "copy_insertion",
    srcs = ["copy_insertion.cc"],
    hdrs = ["copy_insertion.h"],
    deps = [
        ":buffer_liveness",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":liveness_util",
        ":logical_buffer",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "copy_insertion_test",
    srcs = ["copy_insertion_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_dce",
    srcs = ["hlo_dce.cc"],
    hdrs = ["hlo_dce.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_verifier",
    srcs = ["hlo_verifier.cc"],
    hdrs = ["hlo_verifier.h"],
    deps = [
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_verifier_test",
    srcs = ["hlo_verifier_test.cc"],
    deps = [
        ":hlo",
        ":hlo_verifier",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_rematerialization",
    srcs = ["hlo_rematerialization.cc"],
    hdrs = ["hlo_rematerialization.h"],
    deps = [
        ":buffer_liveness",
        ":call_graph",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_dce",
        ":hlo_ordering",
        ":hlo_scheduling",
        ":liveness_util",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_rematerialization_test",
    srcs = ["hlo_rematerialization_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "hlo_dce_test",
    srcs = ["hlo_dce_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "layout_assignment_test",
    srcs = ["layout_assignment_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":computation_layout",
        ":hlo",
        ":hlo_matchers",
        ":layout_assignment",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tools/parser:hlo_parser",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_pass",
    hdrs = [
        "hlo_pass_fix.h",
        "hlo_pass_interface.h",
    ],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_pass_pipeline",
    srcs = [
        "hlo_pass_pipeline.cc",
    ],
    hdrs = [
        "hlo_pass_pipeline.h",
    ],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_proto_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_cse",
    srcs = ["hlo_cse.cc"],
    hdrs = ["hlo_cse.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_cse_test",
    srcs = ["hlo_cse_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo",
        ":hlo_cse",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_constant_folding",
    srcs = ["hlo_constant_folding.cc"],
    hdrs = ["hlo_constant_folding.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_constant_folding_test",
    srcs = ["hlo_constant_folding_test.cc"],
    deps = [
        ":hlo",
        ":hlo_constant_folding",
        ":hlo_matchers",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_element_type_converter",
    srcs = ["hlo_element_type_converter.cc"],
    hdrs = ["hlo_element_type_converter.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_element_type_converter_test",
    srcs = ["hlo_element_type_converter_test.cc"],
    deps = [
        ":hlo_element_type_converter",
        ":hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "device_memory_allocator",
    srcs = ["device_memory_allocator.cc"],
    hdrs = ["device_memory_allocator.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":hlo",
        ":hlo_module_config",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@llvm//:core",
        "@llvm//:transform_utils",
    ],
)

cc_library(
    name = "hlo_module_config",
    srcs = ["hlo_module_config.cc"],
    hdrs = ["hlo_module_config.h"],
    deps = [
        ":computation_layout",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "computation_layout",
    srcs = ["computation_layout.cc"],
    hdrs = ["computation_layout.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_subcomputation_unification",
    srcs = ["hlo_subcomputation_unification.cc"],
    hdrs = ["hlo_subcomputation_unification.h"],
    deps = [
        ":hlo_pass",
    ],
)

tf_cc_test(
    name = "hlo_subcomputation_unification_test",
    srcs = ["hlo_subcomputation_unification_test.cc"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_subcomputation_unification",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_tfgraph_builder",
    srcs = ["hlo_tfgraph_builder.cc"],
    hdrs = ["hlo_tfgraph_builder.h"],
    visibility = ["//tensorflow/compiler/xla/tools:__pkg__"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "hlo_tfgraph_builder_test",
    srcs = ["hlo_tfgraph_builder_test.cc"],
    deps = [
        ":hlo_tfgraph_builder",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "hlo_graph_dumper",
    srcs = [
        "hlo_graph_dumper.cc",
    ],
    hdrs = ["hlo_graph_dumper.h"],
    deps = [
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_tfgraph_builder",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:regexp_internal",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "hlo_graph_dumper_test",
    srcs = ["hlo_graph_dumper_test.cc"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "transpose_folding",
    srcs = ["transpose_folding.cc"],
    hdrs = ["transpose_folding.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "transpose_folding_test",
    srcs = ["transpose_folding_test.cc"],
    deps = [
        ":hlo",
        ":shape_inference",
        ":transpose_folding",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "zero_sized_hlo_elimination",
    srcs = ["zero_sized_hlo_elimination.cc"],
    hdrs = ["zero_sized_hlo_elimination.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "zero_sized_hlo_elimination_test",
    srcs = ["zero_sized_hlo_elimination_test.cc"],
    deps = [
        ":hlo",
        ":shape_inference",
        ":zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "pool",
    hdrs = ["pool.h"],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "pool_test",
    srcs = ["pool_test.cc"],
    deps = [
        ":pool",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_proto_util",
    srcs = ["hlo_proto_util.cc"],
    hdrs = ["hlo_proto_util.h"],
    deps = [
        ":buffer_assignment",
        ":hlo",
        ":hlo_proto",
        "//tensorflow/compiler/xla:status",
    ],
)

cc_library(
    name = "reduce_precision_insertion",
    srcs = ["reduce_precision_insertion.cc"],
    hdrs = ["reduce_precision_insertion.h"],
    deps = [
        ":buffer_liveness",
        ":hlo",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "reduce_precision_insertion_test",
    size = "small",
    srcs = ["reduce_precision_insertion_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":reduce_precision_insertion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_runner",
    srcs = ["hlo_runner.cc"],
    hdrs = ["hlo_runner.h"],
    deps = [
        ":executable",
        ":hlo",
        ":transfer_manager",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/tools/parser:hlo_parser",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "hlo_profile_printer",
    srcs = ["hlo_profile_printer.cc"],
    hdrs = ["hlo_profile_printer.h"],
    deps = [
        ":hlo_profile_printer_data",
        ":human_readable_profile_builder",
        "//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "tuple_util",
    srcs = ["tuple_util.cc"],
    hdrs = ["tuple_util.h"],
    deps = [
        ":hlo",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tuple_util_test",
    srcs = ["tuple_util_test.cc"],
    deps = [
        ":tuple_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/compiler/xla/tools/parser:hlo_parser",
    ],
)

cc_library(
    name = "while_util",
    srcs = ["while_util.cc"],
    hdrs = ["while_util.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":tuple_util",
    ],
)

tf_cc_test(
    name = "while_util_test",
    srcs = ["while_util_test.cc"],
    deps = [
        ":while_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/compiler/xla/tools/parser:hlo_parser",
    ],
)

cc_library(
    name = "while_loop_invariant_code_motion",
    srcs = ["while_loop_invariant_code_motion.cc"],
    hdrs = ["while_loop_invariant_code_motion.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":tuple_util",
        ":while_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "while_loop_invariant_code_motion_test",
    srcs = ["while_loop_invariant_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_loop_invariant_code_motion",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_verified_test_base",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "source_map_util",
    srcs = ["source_map_util.cc"],
    hdrs = ["source_map_util.h"],
    deps = [
        ":executable",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
