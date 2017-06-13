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

xla_proto_library(
    name = "session_proto",
    srcs = ["session.proto"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

xla_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto",
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
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
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

cc_test(
    name = "shape_inference_test",
    size = "small",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "hlo_opcode_test",
    size = "small",
    srcs = ["hlo_opcode_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "hlo_evaluator",
    srcs = ["hlo_evaluator.cc"],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        ":hlo",
        ":hlo_query",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "hlo_evaluator_test",
    size = "small",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
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
    ],
    hdrs = [
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "hlo_computation.h",
        "hlo_instruction.h",
        "hlo_module.h",
        "hlo_opcode.h",
    ],
    deps = [
        ":hlo_module_config",
        ":hlo_proto",
        ":name_uniquer",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:protobuf_util",
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
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "hlo_matchers_test",
    size = "small",
    srcs = ["hlo_matchers_test.cc"],
    deps = [
        ":hlo_matchers",
        "//tensorflow/compiler/xla:shape_util",
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

cc_test(
    name = "hlo_instruction_test",
    size = "small",
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

cc_test(
    name = "call_graph_test",
    size = "small",
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

cc_test(
    name = "flatten_call_graph_test",
    size = "small",
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
        "//tensorflow/compiler/xla/legacy_flags:user_computation_flags",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "user_computation_test",
    size = "small",
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
        "//tensorflow/compiler/xla/legacy_flags:user_computation_flags",
        "//tensorflow/compiler/xla/service:hlo",
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
        ":device_memory_allocator",
        ":platform_util",
        ":pool",
        ":transfer_manager",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/legacy_flags:backend_flags",
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
        ":cpu_transfer_manager",
        ":device_memory_allocator",
        ":executable",
        ":execution_tracker",
        ":gpu_transfer_manager",
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":platform_util",
        ":session_proto",
        ":transfer_manager",
        ":user_computation",
        ":versioned_computation_handle",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/legacy_flags:service_flags",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
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
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:service_flags",
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
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "cpu_plugin",
    deps = [
        ":cpu_transfer_manager",
        ":service",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "gpu_plugin",
    deps = [
        ":gpu_transfer_manager",
        ":service",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",
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
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:service_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
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
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
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

cc_library(
    name = "liveness_util",
    srcs = ["liveness_util.cc"],
    hdrs = ["liveness_util.h"],
    deps = [
        ":hlo",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
    ],
)

cc_test(
    name = "liveness_util_test",
    size = "small",
    srcs = ["liveness_util_test.cc"],
    deps = [
        ":hlo",
        ":liveness_util",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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

cc_test(
    name = "buffer_liveness_test",
    size = "small",
    srcs = ["buffer_liveness_test.cc"],
    deps = [
        ":buffer_liveness",
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
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
        ":hlo",
        ":hlo_ordering",
        ":hlo_proto",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/legacy_flags:buffer_assignment_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_test(
    name = "buffer_assignment_test",
    size = "small",
    srcs = ["buffer_assignment_test.cc"],
    deps = [
        ":buffer_assignment",
        ":call_graph",
        ":computation_tracker",
        ":copy_insertion",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "heap_simulator_test",
    size = "small",
    srcs = ["heap_simulator_test.cc"],
    deps = [
        ":hlo",
        ":hlo_ordering",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

# The hlo_ordering library contains both hlo_ordering and heap_simulator because
# they are mutually dependent.
cc_library(
    name = "hlo_ordering",
    srcs = [
        "heap_simulator.cc",
        "hlo_ordering.cc",
    ],
    hdrs = [
        "heap_simulator.h",
        "hlo_ordering.h",
    ],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_proto",
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

cc_test(
    name = "hlo_ordering_test",
    size = "small",
    srcs = ["hlo_ordering_test.cc"],
    deps = [
        ":hlo",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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

cc_test(
    name = "instruction_fusion_test",
    size = "small",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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

cc_test(
    name = "algebraic_simplifier_test",
    size = "small",
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
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
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

cc_test(
    name = "reshape_mover_test",
    size = "small",
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
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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

cc_test(
    name = "inliner_test",
    size = "small",
    srcs = ["inliner_test.cc"],
    deps = [
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
        "//tensorflow/core:test_main",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "cpu_transfer_manager",
    srcs = ["cpu_transfer_manager.cc"],
    hdrs = ["cpu_transfer_manager.h"],
    deps = [
        ":generic_transfer_manager",
        ":transfer_manager",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_transfer_manager",
    srcs = ["gpu_transfer_manager.cc"],
    hdrs = ["gpu_transfer_manager.h"],
    deps = [
        ":generic_transfer_manager",
        ":transfer_manager",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/gpu:infeed_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_test(
    name = "transfer_manager_test",
    size = "small",
    srcs = ["transfer_manager_test.cc"],
    deps = [
        ":cpu_transfer_manager",
        ":generic_transfer_manager",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "hlo_cost_analysis",
    srcs = [
        "hlo_cost_analysis.cc",
    ],
    hdrs = [
        "hlo_cost_analysis.h",
    ],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "hlo_cost_analysis_test",
    size = "small",
    srcs = ["hlo_cost_analysis_test.cc"],
    deps = [
        ":computation_tracker",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "hlo_execution_profile",
    srcs = ["hlo_execution_profile.cc"],
    hdrs = ["hlo_execution_profile.h"],
    deps = [
        ":hlo",
        ":hlo_cost_analysis",
        "//tensorflow/compiler/xla:metric_table_report",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_test(
    name = "hlo_computation_test",
    size = "small",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_binary(
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

cc_test(
    name = "hlo_module_test",
    size = "small",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = [
        "logical_buffer.cc",
    ],
    hdrs = [
        "logical_buffer.h",
    ],
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
    name = "hlo_dataflow_analysis",
    srcs = [
        "hlo_dataflow_analysis.cc",
    ],
    hdrs = [
        "hlo_dataflow_analysis.h",
    ],
    deps = [
        ":call_graph",
        ":hlo",
        ":liveness_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_test(
    name = "hlo_dataflow_analysis_test",
    size = "small",
    srcs = ["hlo_dataflow_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_matchers",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "hlo_alias_analysis",
    srcs = [
        "hlo_alias_analysis.cc",
    ],
    hdrs = [
        "hlo_alias_analysis.h",
    ],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":logical_buffer",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "hlo_alias_analysis_test",
    srcs = ["hlo_alias_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_matchers",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "tuple_points_to_analysis",
    srcs = [
        "tuple_points_to_analysis.cc",
    ],
    hdrs = [
        "tuple_points_to_analysis.h",
    ],
    deps = [
        ":hlo",
        ":logical_buffer",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "tuple_points_to_analysis_test",
    size = "small",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "compilation_cache",
    srcs = [
        "compilation_cache.cc",
    ],
    hdrs = [
        "compilation_cache.h",
    ],
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
        ":hlo_pass",
        ":liveness_util",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "copy_insertion_test",
    size = "small",
    srcs = ["copy_insertion_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo",
        ":hlo_matchers",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
    deps = [":hlo_pass"],
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
        ":hlo_cost_analysis",
        ":hlo_dce",
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

cc_test(
    name = "hlo_rematerialization_test",
    size = "small",
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
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "hlo_dce_test",
    size = "small",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_test(
    name = "layout_assignment_test",
    size = "small",
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
        "//tensorflow/core:lib",
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
        ":compiler",
        ":hlo",
        ":hlo_pass",
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

cc_test(
    name = "hlo_cse_test",
    size = "small",
    srcs = ["hlo_cse_test.cc"],
    deps = [
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

cc_test(
    name = "hlo_constant_folding_test",
    size = "small",
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
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
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

cc_test(
    name = "hlo_subcomputation_unification_test",
    size = "small",
    srcs = ["hlo_subcomputation_unification_test.cc"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_subcomputation_unification",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "hlo_tfgraph_builder_test",
    size = "small",
    srcs = ["hlo_tfgraph_builder_test.cc"],
    deps = [
        ":hlo_tfgraph_builder",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla/legacy_flags:hlo_graph_dumper_flags",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
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
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "transpose_folding_test",
    size = "small",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
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

cc_test(
    name = "pool_test",
    size = "small",
    srcs = ["pool_test.cc"],
    deps = [
        ":pool",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/core:test_main",
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
