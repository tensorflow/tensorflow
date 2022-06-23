# Description:
#   XLA service implementation.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_proto_library", "xla_py_test_deps")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//tensorflow:tensorflow.bzl", "if_libtpu", "tf_cc_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "internal_hlo_deps")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "internal_cuda_deps")
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//tensorflow/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

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

tf_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/compiler/xla:xla_data_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hlo_profile_printer_data",
    srcs = ["hlo_profile_printer_data.proto"],
    cc_api_version = 2,
)

tf_proto_library(
    name = "hlo_execution_profile_data",
    srcs = ["hlo_execution_profile_data.proto"],
    cc_api_version = 2,
    protodeps = [":hlo_profile_printer_data"],
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
    name = "reduce_scatter_utils",
    srcs = ["reduce_scatter_utils.cc"],
    hdrs = ["reduce_scatter_utils.h"],
    deps = [
        ":hlo",
    ],
)

cc_library(
    name = "async_collective_creator",
    srcs = ["async_collective_creator.cc"],
    hdrs = ["async_collective_creator.h"],
    deps = [
        ":computation_placer_hdr",
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "async_collective_creator_test",
    srcs = ["async_collective_creator_test.cc"],
    deps = [
        ":async_collective_creator",
        ":hlo",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "async_op_canonicalizer",
    srcs = ["async_op_canonicalizer.cc"],
    hdrs = ["async_op_canonicalizer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
    ],
)

tf_cc_test(
    name = "async_op_canonicalizer_test",
    srcs = ["async_op_canonicalizer_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":hlo_dce",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "all_reduce_key",
    srcs = ["all_reduce_key.cc"],
    hdrs = ["all_reduce_key.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "all_reduce_reassociate",
    srcs = ["all_reduce_reassociate.cc"],
    hdrs = ["all_reduce_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_test(
    name = "all_reduce_reassociate_test",
    srcs = ["all_reduce_reassociate_test.cc"],
    deps = [
        ":all_reduce_reassociate",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "all_reduce_folder",
    srcs = ["all_reduce_folder.cc"],
    hdrs = ["all_reduce_folder.h"],
    deps = [
        ":all_reduce_key",
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "all_reduce_folder_test",
    srcs = ["all_reduce_folder_test.cc"],
    deps = [
        ":all_reduce_folder",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
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
    name = "broadcast_canonicalizer",
    srcs = ["broadcast_canonicalizer.cc"],
    hdrs = ["broadcast_canonicalizer.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
    ],
)

tf_cc_test(
    name = "broadcast_canonicalizer_test",
    srcs = ["broadcast_canonicalizer_test.cc"],
    deps = [
        ":broadcast_canonicalizer",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "bfloat16_conversion_folding",
    srcs = ["bfloat16_conversion_folding.cc"],
    hdrs = ["bfloat16_conversion_folding.h"],
    deps = [
        ":bfloat16_support",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "bfloat16_normalization_test",
    srcs = ["bfloat16_normalization_test.cc"],
    deps = [
        ":bfloat16_normalization",
        ":bfloat16_support",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_verifier",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "bfloat16_propagation_test",
    srcs = ["bfloat16_propagation_test.cc"],
    deps = [
        ":bfloat16_propagation",
        ":bfloat16_support",
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_proto_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":hlo",
        ":shape_inference",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
    srcs = [
        "hlo_evaluator.cc",
        "hlo_evaluator_typed_visitor.h",
        "hlo_evaluator_typed_visitor_bfloat16.cc",
        "hlo_evaluator_typed_visitor_bool.cc",
        "hlo_evaluator_typed_visitor_complex128.cc",
        "hlo_evaluator_typed_visitor_complex64.cc",
        "hlo_evaluator_typed_visitor_double.cc",
        "hlo_evaluator_typed_visitor_float.cc",
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        ":dynamic_dimension_inference",
        ":hlo",
        ":hlo_query",
        ":pattern_matcher",
        ":shape_inference",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_live_range",
    srcs = [
        "hlo_live_range.cc",
    ],
    hdrs = [
        "hlo_live_range.h",
    ],
    deps = [
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_live_range_test",
    srcs = ["hlo_live_range_test.cc"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_live_range",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        ":hlo_value",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo",
        ":hlo_element_type_converter",
        ":hlo_evaluator",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo",
    srcs = [
        "dfs_hlo_visitor.cc",
        "dynamic_parameter_binding.cc",
        "hlo_computation.cc",
        "hlo_input_output_alias_config.cc",
        "hlo_instruction.cc",
        "hlo_instructions.cc",
        "hlo_module.cc",
        "hlo_module_metadata.cc",
        "hlo_op_metadata.cc",
        "hlo_opcode.cc",
        "hlo_schedule.cc",
        "hlo_sharding.cc",
        "hlo_sharding_metadata.cc",
    ],
    hdrs = [
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "dynamic_parameter_binding.h",
        "hlo_casting_utils.h",
        "hlo_clone_context.h",
        "hlo_computation.h",
        "hlo_domain_metadata.h",
        "hlo_input_output_alias_config.h",
        "hlo_instruction.h",
        "hlo_instructions.h",
        "hlo_module.h",
        "hlo_module_metadata.h",
        "hlo_op_metadata.h",
        "hlo_opcode.h",
        "hlo_schedule.h",
        "hlo_sharding.h",
        "hlo_sharding_metadata.h",
    ],
    deps = [
        ":compilation_environments",
        ":computation_placer_hdr",
        ":hlo_module_config",
        ":hlo_proto_cc",
        ":mapped_ptr_container_sorter",
        ":name_uniquer",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:human_readable_json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_sharding_util",
    srcs = [
        "hlo_sharding_util.cc",
    ],
    hdrs = [
        "hlo_sharding_util.h",
    ],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_cc_test(
    name = "hlo_sharding_util_test",
    srcs = [
        "hlo_sharding_util_test.cc",
    ],
    deps = [
        ":hlo_sharding_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "sharding_propagation",
    srcs = [
        "sharding_propagation.cc",
    ],
    hdrs = [
        "sharding_propagation.h",
    ],
    deps = [
        ":dot_as_convolution_util",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_sharding_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:sharding_op_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "sharding_propagation_test",
    srcs = [
        "sharding_propagation_test.cc",
    ],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":sharding_propagation",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sharding_remover",
    srcs = [
        "sharding_remover.cc",
    ],
    hdrs = [
        "sharding_remover.h",
    ],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "sharding_remover_test",
    size = "small",
    srcs = [
        "sharding_remover_test.cc",
    ],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":sharding_remover",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "dot_as_convolution_util",
    srcs = [
        "dot_as_convolution_util.cc",
    ],
    hdrs = [
        "dot_as_convolution_util.h",
    ],
    deps = [
        ":hlo",
        ":shape_inference",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
    ],
)

tf_cc_test(
    name = "dynamic_parameter_binding_test",
    srcs = ["dynamic_parameter_binding_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_test(
    name = "dynamic_update_slice_test",
    srcs = ["dynamic_update_slice_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":hlo_parser",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service/cpu:cpu_executable",
        "//tensorflow/compiler/xla/service/cpu:parallel_task_assignment",
        "//tensorflow/compiler/xla/service/cpu:target_machine_features",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "dfs_hlo_visitor_with_default_test",
    srcs = ["dfs_hlo_visitor_with_default_test.cc"],
    deps = [
        ":hlo",
        ":hlo_runner",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "pattern_matcher",
    hdrs = ["pattern_matcher.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
    ],
)

tf_cc_test(
    name = "pattern_matcher_test",
    srcs = ["pattern_matcher_test.cc"],
    deps = [
        ":hlo",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pattern_matcher_gmock",
    testonly = 1,
    hdrs = ["pattern_matcher_gmock.h"],
    deps = [
        ":pattern_matcher",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "pattern_matcher_gmock_test",
    srcs = ["pattern_matcher_gmock_test.cc"],
    deps = [
        ":hlo",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_reachability",
    srcs = ["hlo_reachability.cc"],
    hdrs = ["hlo_reachability.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_reachability_test",
    srcs = ["hlo_reachability_test.cc"],
    deps = [
        ":computation_placer",
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
        ":hlo_parser",
        "//tensorflow/compiler/xla:test",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_matchers_test",
    srcs = ["hlo_matchers_test.cc"],
    deps = [
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "hlo_instruction_test",
    srcs = ["hlo_instruction_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "hlo_sharding_test",
    srcs = ["hlo_sharding_test.cc"],
    deps = [
        ":hlo",
        ":hlo_parser",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":call_graph",
        ":hlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_set",
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
        ":hlo",
        ":hlo_dce",
        ":hlo_domain_isolator",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "flatten_call_graph_test",
    srcs = ["flatten_call_graph_test.cc"],
    deps = [
        ":call_graph",
        ":flatten_call_graph",
        ":hlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":compiler",
        ":computation_placer",
        ":platform_util",
        ":stream_pool",
        ":transfer_manager",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        ":computation_placer",
        ":dump",
        ":dynamic_dimension_inference",
        ":dynamic_padder",
        ":executable",
        ":execution_tracker",
        ":hlo",
        ":hlo_cost_analysis",
        ":hlo_evaluator",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":hlo_proto_util",
        ":platform_util",
        ":source_map_util",
        ":stream_pool",
        ":transfer_manager",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        ":executable",
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":platform_util",
        ":service",
        ":shaped_buffer",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        ":dump",
        ":platform_util",
        ":service",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cpu_plugin",
    compatible_with = [],
    deps = [
        ":service",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_transfer_manager",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "gpu_plugin_impl",
    compatible_with = [],
    deps = if_gpu_is_configured([
        ":service",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_transfer_manager",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ]) + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/service/gpu:amdgpu_compiler",
        "//tensorflow/core/platform/default/build_config:stream_executor_rocm",
    ]) + internal_cuda_deps(),
)

cc_library(
    name = "gpu_plugin_stub",
)

alias(
    name = "gpu_plugin_noncuda",
    actual = if_libtpu("gpu_plugin_stub", "gpu_plugin_impl"),
)

alias(
    name = "gpu_plugin",
    actual = if_cuda("gpu_plugin_impl", "gpu_plugin_noncuda"),
)

cc_library(
    name = "interpreter_plugin",
    deps = [
        ":service",
        "//tensorflow/compiler/xla/service/interpreter:compiler",
        "//tensorflow/compiler/xla/service/interpreter:interpreter_transfer_manager",
        "//tensorflow/compiler/xla/service/interpreter:platform",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "shaped_buffer",
    srcs = ["shaped_buffer.cc"],
    hdrs = ["shaped_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "shaped_buffer_test",
    srcs = ["shaped_buffer_test.cc"],
    deps = [
        ":cpu_plugin",
        ":platform_util",
        ":shaped_buffer",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
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
        ":dump",
        ":hlo",
        ":hlo_execution_profile",
        ":hlo_graph_dumper",
        ":hlo_proto_cc",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        ":stream_pool",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:device_description",
        "//tensorflow/stream_executor:device_memory_allocator",
    ] + internal_hlo_deps(),
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":buffer_assignment",
        ":buffer_value",
        ":computation_placer",
        ":executable",
        ":hlo",
        ":hlo_module_config",
        ":hlo_module_group",
        ":logical_buffer",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        ":compiler",
        "//tensorflow/core:lib_internal",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
        ":compiler",
        ":executable",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "allocation_tracker",
    srcs = ["allocation_tracker.cc"],
    hdrs = ["allocation_tracker.h"],
    deps = [
        ":backend",
        ":transfer_manager",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "execution_tracker",
    srcs = ["execution_tracker.cc"],
    hdrs = ["execution_tracker.h"],
    deps = [
        ":backend",
        ":stream_pool",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "channel_tracker",
    srcs = ["channel_tracker.cc"],
    hdrs = ["channel_tracker.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
    name = "buffer_assignment",
    srcs = [
        "buffer_assignment.cc",
    ],
    hdrs = [
        "buffer_assignment.h",
    ],
    deps = [
        ":buffer_value_containers",
        ":heap_simulator",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_proto_cc",
        ":hlo_value",
        ":logical_buffer",
        ":memory_space_assignment",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "buffer_assignment_test",
    srcs = ["buffer_assignment_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":buffer_assignment",
        ":buffer_value",
        ":call_graph",
        ":copy_insertion",
        ":cpu_plugin",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        ":hlo_proto_cc",
        ":hlo_reachability",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "hlo_ordering_test",
    size = "small",
    srcs = ["hlo_ordering_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "heap_simulator",
    srcs = ["heap_simulator.cc"],
    hdrs = ["heap_simulator.h"],
    deps = [
        ":buffer_value",
        ":buffer_value_containers",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_ordering",
        ":hlo_proto_cc",
        ":memory_space_assignment_repacking",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "heap_simulator_test",
    srcs = ["heap_simulator_test.cc"],
    deps = [
        ":buffer_value",
        ":heap_simulator",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_module_group",
    srcs = ["hlo_module_group.cc"],
    hdrs = ["hlo_module_group.h"],
    deps = [
        ":hlo",
        ":hlo_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_module_group_test",
    srcs = ["hlo_module_group_test.cc"],
    # TODO(b/148211710) Test fails in OSS.
    tags = ["no_oss"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_module_group",
        ":hlo_module_group_metadata",
        ":hlo_proto_cc",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_module_group_metadata",
    srcs = ["hlo_module_group_metadata.cc"],
    hdrs = ["hlo_module_group_metadata.h"],
    deps = [
        ":hlo",
        ":hlo_alias_analysis",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_module_util",
    srcs = ["hlo_module_util.cc"],
    hdrs = ["hlo_module_util.h"],
    deps = [
        ":compiler",
        ":hlo_module_config",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_module_group_util",
    srcs = ["hlo_module_group_util.cc"],
    hdrs = ["hlo_module_group_util.h"],
    deps = [
        ":hlo",
        ":hlo_module_group_metadata",
        ":hlo_reachability",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_schedule_test",
    srcs = ["hlo_schedule_test.cc"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "hlo_input_output_alias_config_test",
    srcs = ["hlo_input_output_alias_config_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "hlo_memory_scheduler",
    srcs = ["hlo_memory_scheduler.cc"],
    hdrs = ["hlo_memory_scheduler.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "hlo_memory_scheduler_test",
    srcs = ["hlo_memory_scheduler_test.cc"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "fusion_queue",
    hdrs = ["fusion_queue.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":fusion_queue",
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_reachability",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_reachability",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_creation_utils",
    srcs = ["hlo_creation_utils.cc"],
    hdrs = [
        "hlo_creation_utils.h",
    ],
    deps = [
        ":hlo",
        ":hlo_module_config",
        ":shape_inference",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fusion_node_indexing_evaluation",
    srcs = ["fusion_node_indexing_evaluation.cc"],
    hdrs = ["fusion_node_indexing_evaluation.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "fusion_node_indexing_evaluation_test",
    srcs = ["fusion_node_indexing_evaluation_test.cc"],
    deps = [
        ":fusion_node_indexing_evaluation",
        ":hlo",
        ":hlo_parser",
        ":instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "hlo_creation_utils_test",
    srcs = ["hlo_creation_utils_test.cc"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_evaluator",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "batchnorm_expander",
    srcs = ["batchnorm_expander.cc"],
    hdrs = ["batchnorm_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "op_expander_pass",
    srcs = ["op_expander_pass.cc"],
    hdrs = ["op_expander_pass.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "gather_expander",
    srcs = ["gather_expander.cc"],
    hdrs = ["gather_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        ":while_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "optimization_barrier_expander",
    srcs = ["optimization_barrier_expander.cc"],
    hdrs = ["optimization_barrier_expander.h"],
    deps = [
        ":op_expander_pass",
    ],
)

cc_library(
    name = "comparison_expander",
    srcs = ["comparison_expander.cc"],
    hdrs = ["comparison_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client/lib:comparators",
    ],
)

cc_library(
    name = "scatter_expander",
    srcs = ["scatter_expander.cc"],
    hdrs = ["scatter_expander.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_creation_utils",
        ":op_expander_pass",
        ":while_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "scatter_expander_test",
    srcs = ["scatter_expander_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":scatter_expander",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "triangular_solve_expander",
    srcs = ["triangular_solve_expander.cc"],
    hdrs = ["triangular_solve_expander.h"],
    deps = [
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "triangular_solve_expander_test",
    size = "medium",
    srcs = ["triangular_solve_expander_test.cc"],
    shard_count = 3,
    deps = [
        ":hlo",
        ":triangular_solve_expander",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "cholesky_expander",
    srcs = ["cholesky_expander.cc"],
    hdrs = ["cholesky_expander.h"],
    deps = [
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:loops",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "qr_expander",
    srcs = ["qr_expander.cc"],
    hdrs = ["qr_expander.h"],
    deps = [
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:loops",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "real_imag_expander",
    srcs = ["real_imag_expander.cc"],
    hdrs = ["real_imag_expander.h"],
    deps = [
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
    ],
)

tf_cc_test(
    name = "real_imag_expander_test",
    size = "small",
    srcs = ["real_imag_expander_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":real_imag_expander",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["eigh_expander.cc"],
    hdrs = ["eigh_expander.h"],
    deps = [
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:loops",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "convolution_4d_expander",
    srcs = ["convolution_4d_expander.cc"],
    hdrs = ["convolution_4d_expander.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "convolution_4d_expander_test",
    srcs = ["convolution_4d_expander_test.cc"],
    deps = [
        "convolution_4d_expander",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
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
        ":hlo_parser",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "algebraic_simplifier",
    srcs = ["algebraic_simplifier.cc"],
    hdrs = ["algebraic_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_evaluator",
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_comparison",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["tree_reduction_rewriter.cc"],
    hdrs = ["tree_reduction_rewriter.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_evaluator",
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "algebraic_simplifier_test",
    srcs = ["algebraic_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "simplify_fp_conversions",
    srcs = ["simplify_fp_conversions.cc"],
    hdrs = ["simplify_fp_conversions.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplify_fp_conversions_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":simplify_fp_conversions",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["logistic_expander.cc"],
    hdrs = ["logistic_expander.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "logistic_expander_test",
    srcs = ["logistic_expander_test.cc"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":logistic_expander",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "collectives_schedule_linearizer",
    srcs = ["collectives_schedule_linearizer.cc"],
    hdrs = ["collectives_schedule_linearizer.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "collectives_schedule_linearizer_test",
    srcs = ["collectives_schedule_linearizer_test.cc"],
    deps = [
        ":collectives_schedule_linearizer",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "collective_combiner_utils",
    hdrs = ["collective_combiner_utils.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_reachability",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:tflite_portable_logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "collective_decomposer_utils",
    srcs = ["collective_decomposer_utils.cc"],
    hdrs = ["collective_decomposer_utils.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo",
        ":hlo_module_config",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "all_gather_broadcast_reorder",
    srcs = ["all_gather_broadcast_reorder.cc"],
    hdrs = ["all_gather_broadcast_reorder.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["bitcast_dtypes_expander.cc"],
    hdrs = ["bitcast_dtypes_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:broadcast",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "bitcast_dtypes_expander_test",
    srcs = ["bitcast_dtypes_expander_test.cc"],
    deps = [
        ":bitcast_dtypes_expander",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:statusor",
    ],
)

cc_library(
    name = "bitcast_decomposer",
    srcs = ["bitcast_decomposer.cc"],
    hdrs = ["bitcast_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
    ],
)

xla_test(
    name = "bitcast_decomposer_test",
    srcs = ["bitcast_decomposer_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":bitcast_decomposer",
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "all_gather_broadcast_reorder_test",
    srcs = ["all_gather_broadcast_reorder_test.cc"],
    deps = [
        ":all_gather_broadcast_reorder",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:statusor",
    ],
)

cc_library(
    name = "all_gather_combiner",
    srcs = ["all_gather_combiner.cc"],
    hdrs = ["all_gather_combiner.h"],
    deps = [
        ":collective_combiner_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "all_gather_combiner_test",
    srcs = ["all_gather_combiner_test.cc"],
    deps = [
        ":all_gather_combiner",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "all_reduce_combiner",
    srcs = ["all_reduce_combiner.cc"],
    hdrs = ["all_reduce_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "all_reduce_combiner_test",
    srcs = ["all_reduce_combiner_test.cc"],
    deps = [
        ":all_reduce_combiner",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "all_reduce_contiguous",
    srcs = ["all_reduce_contiguous.cc"],
    hdrs = ["all_reduce_contiguous.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
    ],
)

tf_cc_test(
    name = "all_reduce_contiguous_test",
    srcs = ["all_reduce_contiguous_test.cc"],
    deps = [
        ":all_reduce_contiguous",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "reduce_scatter_combiner",
    srcs = ["reduce_scatter_combiner.cc"],
    hdrs = ["reduce_scatter_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "reduce_scatter_combiner_test",
    srcs = ["reduce_scatter_combiner_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":reduce_scatter_combiner",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "all_reduce_simplifier",
    srcs = ["all_reduce_simplifier.cc"],
    hdrs = ["all_reduce_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_replication_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
    ],
)

tf_cc_test(
    name = "all_reduce_simplifier_test",
    srcs = ["all_reduce_simplifier_test.cc"],
    deps = [
        ":all_reduce_simplifier",
        ":hlo",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduce_scatter_decomposer",
    srcs = ["reduce_scatter_decomposer.cc"],
    hdrs = ["reduce_scatter_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_module_config",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "reduce_scatter_decomposer_test",
    srcs = ["reduce_scatter_decomposer_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":reduce_scatter_decomposer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "batch_dot_simplification",
    srcs = ["batch_dot_simplification.cc"],
    hdrs = ["batch_dot_simplification.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "batch_dot_simplification_test",
    srcs = ["batch_dot_simplification_test.cc"],
    deps = [
        ":batch_dot_simplification",
        ":hlo",
        ":hlo_matchers",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "gather_expander_test",
    srcs = ["gather_expander_test.cc"],
    deps = [
        ":gather_expander",
        ":hlo_query",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "conditional_simplifier",
    srcs = ["conditional_simplifier.cc"],
    hdrs = ["conditional_simplifier.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "conditional_simplifier_test",
    srcs = ["conditional_simplifier_test.cc"],
    deps = [
        ":conditional_simplifier",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "conditional_code_motion",
    srcs = ["conditional_code_motion.cc"],
    hdrs = ["conditional_code_motion.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo",
        ":hlo_cse",
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":hlo_verifier",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "conditional_code_motion_test",
    srcs = ["conditional_code_motion_test.cc"],
    deps = [
        ":conditional_code_motion",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_library(
    name = "convolution_group_converter",
    srcs = ["convolution_group_converter.cc"],
    hdrs = ["convolution_group_converter.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "convolution_group_converter_test",
    size = "small",
    srcs = ["convolution_group_converter_test.cc"],
    deps = [
        ":convolution_group_converter",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "space_to_batch_converter",
    srcs = ["space_to_batch_converter.cc"],
    hdrs = ["space_to_batch_converter.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "space_to_batch_converter_test",
    size = "small",
    srcs = ["space_to_batch_converter_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":space_to_batch_converter",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "while_loop_analysis",
    srcs = ["while_loop_analysis.cc"],
    hdrs = ["while_loop_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_reachability",
        ":pattern_matcher",
        "@com_google_absl//absl/base",
    ],
)

tf_cc_test(
    name = "while_loop_analysis_test",
    srcs = ["while_loop_analysis_test.cc"],
    deps = [
        ":hlo_parser",
        ":while_loop_analysis",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
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
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        ":while_loop_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:union_find",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "while_loop_simplifier_test",
    srcs = ["while_loop_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo",
        ":hlo_cse",
        ":hlo_dce",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":tuple_simplifier",
        ":while_loop_simplifier",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "while_loop_trip_count_annotator",
    srcs = ["while_loop_trip_count_annotator.cc"],
    hdrs = ["while_loop_trip_count_annotator.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":while_loop_analysis",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
    ],
)

tf_cc_test(
    name = "while_loop_trip_count_annotator_test",
    srcs = ["while_loop_trip_count_annotator_test.cc"],
    deps = [
        ":pattern_matcher",
        ":while_loop_simplifier",
        ":while_loop_trip_count_annotator",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "defuser_test",
    srcs = ["defuser_test.cc"],
    deps = [
        ":defuser",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["dot_decomposer.cc"],
    hdrs = ["dot_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dot_decomposer_test",
    srcs = ["dot_decomposer_test.cc"],
    deps = [
        ":dot_decomposer",
        ":hlo_matchers",
        ":hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "dot_merger",
    srcs = ["dot_merger.cc"],
    hdrs = ["dot_merger.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/compiler/xla/service/graphcycles",
    ],
)

tf_cc_test(
    name = "dot_merger_test",
    srcs = ["dot_merger_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":dot_merger",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "all_to_all_decomposer",
    srcs = ["all_to_all_decomposer.cc"],
    hdrs = ["all_to_all_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "all_gather_decomposer",
    srcs = ["all_gather_decomposer.cc"],
    hdrs = ["all_gather_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "all_gather_decomposer_test",
    srcs = ["all_gather_decomposer_test.cc"],
    deps = [
        ":all_gather_decomposer",
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "reshape_mover",
    srcs = ["reshape_mover.cc"],
    hdrs = ["reshape_mover.h"],
    deps = [
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "reshape_decomposer",
    srcs = ["reshape_decomposer.cc"],
    hdrs = ["reshape_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status",
    ],
)

cc_library(
    name = "reduce_decomposer",
    srcs = ["reduce_decomposer.cc"],
    hdrs = ["reduce_decomposer.h"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status",
    ],
)

tf_cc_test(
    name = "reduce_decomposer_test",
    srcs = ["reduce_decomposer_test.cc"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":reduce_decomposer",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "reshape_decomposer_test",
    srcs = ["reshape_decomposer_test.cc"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":reshape_decomposer",
        ":reshape_mover",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "dynamic_window_utils",
    srcs = ["dynamic_window_utils.cc"],
    hdrs = ["dynamic_window_utils.h"],
    deps = [
        ":hlo",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "dynamic_dimension_inference",
    srcs = ["dynamic_dimension_inference.cc"],
    hdrs = ["dynamic_dimension_inference.h"],
    deps = [
        ":dynamic_window_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":tuple_util",
        ":while_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dynamic_dimension_simplifier",
    srcs = ["dynamic_dimension_simplifier.cc"],
    hdrs = ["dynamic_dimension_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
    ],
)

tf_cc_test(
    name = "dynamic_dimension_simplifier_test",
    srcs = ["dynamic_dimension_simplifier_test.cc"],
    deps = [
        ":dynamic_dimension_simplifier",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dynamic_padder",
    srcs = ["dynamic_padder.cc"],
    hdrs = ["dynamic_padder.h"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_window_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_dce",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_test(
    name = "dynamic_padder_test",
    srcs = ["dynamic_padder_test.cc"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_dimension_simplifier",
        ":dynamic_padder",
        ":hlo",
        ":hlo_dce",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dynamic_dimension_inference_test",
    srcs = ["dynamic_dimension_inference_test.cc"],
    deps = [
        ":dynamic_dimension_inference",
        ":hlo",
        ":hlo_matchers",
        ":hlo_runner",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "reshape_mover_test",
    srcs = ["reshape_mover_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":reshape_mover",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "computation_placer",
    srcs = ["computation_placer.cc"],
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/rocm:rocm_platform_id",
    ] + if_libtpu([":tpu_computation_placer"]),
    alwayslink = True,  # Contains per-platform computation placer registration
)

cc_library(
    name = "computation_placer_hdr",
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "tpu_computation_placer",
    srcs = ["tpu_computation_placer.cc"],
    hdrs = ["tpu_computation_placer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":computation_placer_hdr",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor/tpu:status_helper",
        "//tensorflow/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_platform_hdr",
        "//tensorflow/stream_executor/tpu:tpu_platform_id",
        "//tensorflow/stream_executor/tpu:tpu_topology_external",
    ],
    alwayslink = True,  # Contains TPU computation placer registration
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "generic_transfer_manager",
    srcs = ["generic_transfer_manager.cc"],
    hdrs = ["generic_transfer_manager.h"],
    deps = [
        ":transfer_manager",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
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
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_cost_analysis_test",
    srcs = ["hlo_cost_analysis_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo",
        ":hlo_cost_analysis",
        ":local_service",
        ":service",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
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
        ":hlo_execution_profile_data_cc",
        ":hlo_profile_printer",
        ":human_readable_profile_builder",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_computation_test",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "hlo_module_test",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_memory_scheduler",
        ":test_compilation_environment_proto_cc",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_module_metadata_test",
    srcs = ["hlo_module_metadata_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "buffer_value",
    srcs = ["buffer_value.cc"],
    hdrs = ["buffer_value.h"],
    deps = [
        ":hlo",
        ":hlo_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer_value_containers",
    hdrs = ["buffer_value_containers.h"],
    deps = [
        ":buffer_value",
        ":logical_buffer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = ["logical_buffer.cc"],
    hdrs = ["logical_buffer.h"],
    deps = [
        ":buffer_value",
        ":hlo",
        ":hlo_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_value",
    srcs = ["hlo_value.cc"],
    hdrs = ["hlo_value.h"],
    deps = [
        ":buffer_value",
        ":hlo",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_dataflow_analysis",
    srcs = ["hlo_dataflow_analysis.cc"],
    hdrs = ["hlo_dataflow_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_phi_graph",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_dataflow_analysis_test",
    srcs = ["hlo_dataflow_analysis_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_phi_graph",
    srcs = ["hlo_phi_graph.cc"],
    hdrs = ["hlo_phi_graph.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_phi_graph_test",
    srcs = ["hlo_phi_graph_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_phi_graph",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_replication_analysis",
    srcs = ["hlo_replication_analysis.cc"],
    hdrs = ["hlo_replication_analysis.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_replication_analysis_test",
    srcs = ["hlo_replication_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_replication_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_liveness_analysis",
    srcs = ["hlo_liveness_analysis.cc"],
    hdrs = ["hlo_liveness_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_value",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_liveness_analysis_test",
    srcs = ["hlo_liveness_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_liveness_analysis",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "tuple_points_to_analysis",
    srcs = ["tuple_points_to_analysis.cc"],
    hdrs = ["tuple_points_to_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":logical_buffer",
        ":logical_buffer_analysis",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "tuple_points_to_analysis_test",
    srcs = ["tuple_points_to_analysis_test.cc"],
    deps = [
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":instruction_fusion",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
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
        ":call_graph",
        ":computation_layout",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "copy_insertion",
    srcs = ["copy_insertion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "copy_insertion.h",
    ],
    deps = [
        ":dump",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "loop_schedule_linearizer",
    srcs = ["loop_schedule_linearizer.cc"],
    hdrs = ["loop_schedule_linearizer.h"],
    deps = [
        ":dump",
        ":hlo",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service/graphcycles",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "loop_schedule_linearizer_test",
    srcs = ["loop_schedule_linearizer_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        ":loop_schedule_linearizer",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "memory_space_assignment_utils",
    srcs = ["memory_space_assignment_utils.cc"],
    hdrs = ["memory_space_assignment_utils.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
    ],
)

cc_library(
    name = "memory_space_assignment_tuning_utils",
    srcs = ["memory_space_assignment_tuning_utils.cc"],
    hdrs = ["memory_space_assignment_tuning_utils.h"],
    deps = [
        ":heap_simulator",
        ":hlo",
        ":hlo_module_config",
        ":memory_space_assignment_utils",
    ],
)

cc_library(
    name = "memory_space_assignment_repacking",
    hdrs = ["memory_space_assignment_repacking.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "memory_space_assignment_best_fit_repacker",
    srcs = ["memory_space_assignment_best_fit_repacker.cc"],
    hdrs = ["memory_space_assignment_best_fit_repacker.h"],
    deps = [
        ":heap_simulator",
        ":memory_space_assignment_repacking",
    ],
)

tf_cc_test(
    name = "memory_space_assignment_best_fit_repacker_test",
    srcs = ["memory_space_assignment_best_fit_repacker_test.cc"],
    deps = [
        ":memory_space_assignment_best_fit_repacker",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "memory_space_assignment",
    srcs = ["memory_space_assignment.cc"],
    hdrs = ["memory_space_assignment.h"],
    deps = [
        ":heap_simulator",
        ":hlo_cost_analysis",
        ":memory_space_assignment_repacking",
        ":memory_space_assignment_tuning_utils",
        ":memory_space_assignment_utils",
        ":tuple_util",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/core/lib/math:math_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "memory_space_assignment_test",
    srcs = ["memory_space_assignment_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":memory_space_assignment",
        ":memory_space_assignment_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "memory_space_propagation",
    srcs = ["memory_space_propagation.cc"],
    hdrs = ["memory_space_propagation.h"],
    deps = [
        ":hlo",
        ":hlo_dataflow_analysis",
        ":hlo_pass",
    ],
)

tf_cc_test(
    name = "memory_space_propagation_test",
    srcs = ["memory_space_propagation_test.cc"],
    deps = [
        ":hlo_parser",
        ":memory_space_propagation",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "hlo_module_dce",
    srcs = ["hlo_module_dce.cc"],
    hdrs = ["hlo_module_dce.h"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_liveness_analysis",
        ":hlo_pass",
        ":tuple_simplifier",
        ":while_loop_simplifier",
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
        ":collective_ops_utils",
        ":hlo",
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_verifier_test",
    srcs = ["hlo_verifier_test.cc"],
    deps = [
        ":hlo",
        ":hlo_module_config",
        ":hlo_parser",
        ":hlo_verifier",
        ":layout_assignment",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_rematerialization",
    srcs = ["hlo_rematerialization.cc"],
    hdrs = ["hlo_rematerialization.h"],
    deps = [
        ":buffer_value",
        ":call_graph",
        ":flatten_call_graph",
        ":hlo",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_query",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo_rematerialization_test_utils",
    testonly = 1,
    hdrs = ["hlo_rematerialization_test_utils.h"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/flags:flag",
    ],
)

tf_cc_test(
    name = "hlo_rematerialization_test_utils_test",
    srcs = ["hlo_rematerialization_test_utils_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization_test_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "hlo_rematerialization_test",
    srcs = ["hlo_rematerialization_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization",
        ":hlo_rematerialization_test_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "hlo_dce_test",
    srcs = ["hlo_dce_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "hlo_module_dce_test",
    srcs = ["hlo_module_dce_test.cc"],
    deps = [
        ":hlo",
        ":hlo_module_dce",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
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
        ":dynamic_padder",
        ":hlo",
        ":hlo_parser",
        ":layout_assignment",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:span",
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
        ":hlo_module_group",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "@com_google_absl//absl/container:flat_hash_set",
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
        ":compilation_stats",
        ":dump",
        ":hlo",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_proto_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "hlo_pass_pipeline_test",
    srcs = ["hlo_pass_pipeline_test.cc"],
    deps = [
        ":hlo",
        ":hlo_parser",
        ":hlo_pass_pipeline",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "hlo_cse",
    srcs = ["hlo_cse.cc"],
    hdrs = ["hlo_cse.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
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
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        ":slow_operation_alarm",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "hlo_constant_folding_test",
    srcs = ["hlo_constant_folding_test.cc"],
    deps = [
        ":hlo",
        ":hlo_constant_folding",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_domain_map",
    srcs = ["hlo_domain_map.cc"],
    hdrs = ["hlo_domain_map.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hlo_domain_verifier",
    srcs = ["hlo_domain_verifier.cc"],
    hdrs = ["hlo_domain_verifier.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_domain_isolator",
    srcs = ["hlo_domain_isolator.cc"],
    hdrs = ["hlo_domain_isolator.h"],
    deps = [
        ":hlo",
        ":hlo_domain_remover",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/platform:statusor",
    ],
)

cc_library(
    name = "hlo_domain_remover",
    srcs = ["hlo_domain_remover.cc"],
    hdrs = ["hlo_domain_remover.h"],
    deps = [
        ":hlo",
        ":hlo_domain_map",
        ":hlo_domain_verifier",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "hlo_domain_test",
    srcs = ["hlo_domain_test.cc"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_domain_isolator",
        ":hlo_domain_remover",
        ":hlo_parser",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/xla:literal",
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
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "conditional_canonicalizer",
    srcs = ["conditional_canonicalizer.cc"],
    hdrs = ["conditional_canonicalizer.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
    ],
)

tf_cc_test(
    name = "conditional_canonicalizer_test",
    srcs = ["conditional_canonicalizer_test.cc"],
    deps = [
        ":conditional_canonicalizer",
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "maybe_owning_device_memory",
    srcs = [
        "maybe_owning_device_memory.cc",
    ],
    hdrs = [
        "maybe_owning_device_memory.h",
    ],
    deps = [
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_builder_mixin",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

xla_test(
    name = "elemental_ir_emitter_test",
    srcs = ["elemental_ir_emitter_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = [
        "no_windows",  # TODO(b/152037541)
    ],
    deps = [
        ":hlo_parser",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_header",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_module_config",
    srcs = ["hlo_module_config.cc"],
    hdrs = ["hlo_module_config.h"],
    deps = [
        ":computation_layout",
        ":computation_placer",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "computation_layout",
    srcs = ["computation_layout.cc"],
    hdrs = ["computation_layout.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/strings",
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
    name = "hlo_graph_dumper",
    srcs = ["hlo_graph_dumper.cc"],
    hdrs = ["hlo_graph_dumper.h"],
    deps = [
        ":hlo",
        ":hlo_execution_profile",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/service/gpu:cublas_cudnn",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "hlo_graph_dumper_test",
    srcs = ["hlo_graph_dumper_test.cc"],
    deps = [
        ":hlo",
        ":hlo_graph_dumper",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "transpose_folding_test",
    srcs = ["transpose_folding_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":shape_inference",
        ":transpose_folding",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "zero_sized_hlo_elimination",
    srcs = ["zero_sized_hlo_elimination.cc"],
    hdrs = ["zero_sized_hlo_elimination.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:literal",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "stream_pool",
    srcs = ["stream_pool.cc"],
    hdrs = ["stream_pool.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "stream_pool_test",
    srcs = ["stream_pool_test.cc"],
    deps = [
        ":stream_pool",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "hlo_proto_util",
    srcs = ["hlo_proto_util.cc"],
    hdrs = ["hlo_proto_util.h"],
    deps = [
        ":buffer_assignment",
        ":hlo",
        ":hlo_proto_cc",
        ":hlo_verifier",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
    ],
)

tf_cc_test(
    name = "hlo_proto_util_test",
    srcs = ["hlo_proto_util_test.cc"],
    deps = [
        ":hlo",
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_runner_interface",
    srcs = ["hlo_runner_interface.cc"],
    hdrs = ["hlo_runner_interface.h"],
    deps = [
        ":computation_placer",
        ":executable",
        ":hlo",
        ":hlo_parser",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_runner",
    srcs = ["hlo_runner.cc"],
    hdrs = ["hlo_runner.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_placer",
        ":executable",
        ":hlo",
        ":hlo_module_group",
        ":hlo_module_util",
        ":hlo_parser",
        ":hlo_runner_interface",
        ":transfer_manager",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_profile_printer",
    srcs = ["hlo_profile_printer.cc"],
    hdrs = ["hlo_profile_printer.h"],
    deps = [
        ":hlo_profile_printer_data_cc",
        ":human_readable_profile_builder",
        "//tensorflow/compiler/xla:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sort_simplifier",
    srcs = ["sort_simplifier.cc"],
    hdrs = ["sort_simplifier.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "sort_simplifier_test",
    srcs = ["sort_simplifier_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":sort_simplifier",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["stable_sort_expander.cc"],
    hdrs = ["stable_sort_expander.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "stable_sort_expander_test",
    srcs = ["stable_sort_expander_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":stable_sort_expander",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "tuple_util",
    srcs = ["tuple_util.cc"],
    hdrs = ["tuple_util.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "tuple_util_test",
    srcs = ["tuple_util_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_module_config",
        ":hlo_parser",
        ":tuple_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "root_instruction_sinker",
    srcs = ["root_instruction_sinker.cc"],
    hdrs = ["root_instruction_sinker.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":tuple_util",
    ],
)

tf_cc_test(
    name = "root_instruction_sinker_test",
    srcs = ["root_instruction_sinker_test.cc"],
    deps = [
        ":hlo_matchers",
        ":root_instruction_sinker",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "while_util",
    srcs = ["while_util.cc"],
    hdrs = ["while_util.h"],
    deps = [
        ":call_inliner",
        ":hlo",
        ":hlo_creation_utils",
        ":tuple_util",
        "//tensorflow/compiler/xla:literal_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "while_util_test",
    srcs = ["while_util_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "while_loop_all_reduce_code_motion",
    srcs = ["while_loop_all_reduce_code_motion.cc"],
    hdrs = ["while_loop_all_reduce_code_motion.h"],
    deps = [
        ":call_graph",
        ":collective_ops_utils",
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_replication_analysis",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "while_loop_all_reduce_code_motion_test",
    srcs = ["while_loop_all_reduce_code_motion_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_verifier",
        ":while_loop_all_reduce_code_motion",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "while_loop_concat_code_motion",
    srcs = ["while_loop_concat_code_motion.cc"],
    hdrs = ["while_loop_concat_code_motion.h"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":tuple_simplifier",
        ":while_loop_simplifier",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "while_loop_concat_code_motion_test",
    srcs = ["while_loop_concat_code_motion_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_verifier",
        ":while_loop_concat_code_motion",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "while_loop_invariant_code_motion",
    srcs = ["while_loop_invariant_code_motion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "while_loop_invariant_code_motion.h",
    ],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_util",
        ":while_loop_analysis",
        ":while_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "while_loop_invariant_code_motion_test",
    srcs = ["while_loop_invariant_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":while_loop_invariant_code_motion",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "while_loop_expensive_invariant_code_motion",
    srcs = ["while_loop_expensive_invariant_code_motion.cc"],
    hdrs = ["while_loop_expensive_invariant_code_motion.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":tuple_util",
        ":while_loop_analysis",
        ":while_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_cc_test(
    name = "while_loop_expensive_invariant_code_motion_test",
    srcs = ["while_loop_expensive_invariant_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":while_loop_expensive_invariant_code_motion",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "while_loop_constant_sinking",
    srcs = ["while_loop_constant_sinking.cc"],
    hdrs = ["while_loop_constant_sinking.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":while_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_cc_test(
    name = "while_loop_constant_sinking_test",
    srcs = ["while_loop_constant_sinking_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_loop_constant_sinking",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "despecializer",
    srcs = ["despecializer.cc"],
    hdrs = ["despecializer.h"],
    deps = [
        ":bfloat16_normalization",
        ":defuser",
        ":hlo",
        ":hlo_memory_scheduler",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "source_map_util",
    srcs = [],
    hdrs = ["source_map_util.h"],
    deps = [
        ":executable",
        "//tensorflow/compiler/xla:status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "indexed_array_analysis",
    srcs = ["indexed_array_analysis.cc"],
    hdrs = ["indexed_array_analysis.h"],
    deps = [
        ":hlo",
        ":hlo_evaluator",
        ":hlo_pass",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:ptr_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "indexed_array_analysis_test",
    srcs = ["indexed_array_analysis_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":indexed_array_analysis",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_parser",
    srcs = ["hlo_parser.cc"],
    hdrs = ["hlo_parser.h"],
    deps = [
        ":computation_layout",
        ":hlo",
        ":hlo_lexer",
        ":shape_inference",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "hlo_parser_test",
    size = "small",
    srcs = ["hlo_parser_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = [
        "hlo_lexer.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "map_inliner",
    srcs = ["map_inliner.cc"],
    hdrs = ["map_inliner.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "optimize_input_output_buffer_alias",
    srcs = ["optimize_input_output_buffer_alias.cc"],
    hdrs = ["optimize_input_output_buffer_alias.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "optimize_input_output_buffer_alias_test",
    srcs = ["optimize_input_output_buffer_alias_test.cc"],
    deps = [
        ":optimize_input_output_buffer_alias",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ar_crs_combiner",
    srcs = ["ar_crs_combiner.cc"],
    hdrs = ["ar_crs_combiner.h"],
    deps = [
        ":call_graph",
        ":hlo",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_replication_analysis",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compilation_stats",
    srcs = ["compilation_stats.cc"],
    hdrs = ["compilation_stats.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["dynamic_index_splitter.cc"],
    hdrs = ["dynamic_index_splitter.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dynamic_index_splitter_test",
    srcs = ["dynamic_index_splitter_test.cc"],
    deps = [
        ":dynamic_index_splitter",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "ar_crs_combiner_test",
    srcs = ["ar_crs_combiner_test.cc"],
    deps = [
        ":ar_crs_combiner",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "map_inliner_test",
    srcs = ["map_inliner_test.cc"],
    deps = [
        ":hlo",
        ":hlo_matchers",
        ":map_inliner",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "hlo_casting_utils_test",
    srcs = ["hlo_casting_utils_test.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "conditional_to_select",
    srcs = ["conditional_to_select.cc"],
    hdrs = ["conditional_to_select.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "conditional_to_select_test",
    srcs = ["conditional_to_select_test.cc"],
    deps = [
        ":conditional_to_select",
        ":hlo",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "slice_sinker",
    srcs = ["slice_sinker.cc"],
    hdrs = ["slice_sinker.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "custom_call_target_registry",
    srcs = ["custom_call_target_registry.cc"],
    hdrs = ["custom_call_target_registry.h"],
    visibility = ["//visibility:public"],
)

# Exposes the public interface only and hides internal details. Suitable for
# linking into a static library or binary.
cc_library(
    name = "custom_call_status",
    srcs = [
        "custom_call_status.cc",
        "custom_call_status_internal.h",
    ],
    hdrs = ["custom_call_status.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

# Internal version that exposes internal details and private interfaces. For
# use by other XLA code only.
cc_library(
    name = "custom_call_status_internal",
    hdrs = [
        "custom_call_status_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        ":__subpackages__",
        "//tensorflow/compiler/tf2xla:__pkg__",
    ],
    deps = [
        ":custom_call_status",
        "@com_google_absl//absl/strings",
    ],
)

# Public headers only, suitable for inclusion in a shared library of CustomCall
# target functions.
cc_library(
    name = "custom_call_status_public_headers",
    hdrs = ["custom_call_status.h"],
    visibility = ["//visibility:public"],
)

tf_cc_test(
    name = "custom_call_status_test",
    srcs = ["custom_call_status_test.cc"],
    deps = [
        ":custom_call_status_internal",
        ":custom_call_status_test_c_caller",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "custom_call_status_test_c_caller",
    testonly = True,
    srcs = ["custom_call_status_test_c_caller.c"],
    hdrs = ["custom_call_status_test_c_caller.h"],
    deps = [":custom_call_status"],
)

tf_cc_test(
    name = "slice_sinker_test",
    srcs = ["slice_sinker_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":slice_sinker",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["rng_expander.cc"],
    hdrs = ["rng_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:prng",
    ],
)

cc_library(
    name = "rng_bit_generator_expander",
    srcs = ["rng_bit_generator_expander.cc"],
    hdrs = ["rng_bit_generator_expander.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:prng",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "slow_operation_alarm",
    srcs = ["slow_operation_alarm.cc"],
    hdrs = ["slow_operation_alarm.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "collective_ops_utils",
    srcs = ["collective_ops_utils.cc"],
    hdrs = ["collective_ops_utils.h"],
    deps = [
        ":computation_placer",
        ":global_device_id",
        ":hlo",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib_internal",  # fixdeps: keep
    ],
)

tf_cc_test(
    name = "collective_ops_utils_test",
    srcs = ["collective_ops_utils_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":computation_placer",
        ":global_device_id",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "topk_rewriter",
    srcs = ["topk_rewriter.cc"],
    hdrs = ["topk_rewriter.h"],
    deps = [
        ":hlo",
        ":hlo_pass",
        ":pattern_matcher",
        "//tensorflow/compiler/xla:shape_util",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "topk_rewriter_test",
    srcs = ["topk_rewriter_test.cc"],
    deps = [
        ":hlo",
        ":hlo_dce",
        ":hlo_matchers",
        ":topk_rewriter",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_macros_cpu",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "operand_upcaster",
    srcs = ["operand_upcaster.cc"],
    hdrs = ["operand_upcaster.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        ":shape_inference",
    ],
)

tf_cc_test(
    name = "operand_upcaster_test",
    srcs = ["operand_upcaster_test.cc"],
    deps = [
        ":hlo_matchers",
        ":operand_upcaster",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "result_caster",
    srcs = ["result_caster.cc"],
    hdrs = ["result_caster.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        ":shape_inference",
    ],
)

tf_cc_test(
    name = "result_caster_test",
    srcs = ["result_caster_test.cc"],
    deps = [
        ":hlo_matchers",
        ":result_caster",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "global_device_id",
    srcs = ["global_device_id.cc"],
    hdrs = ["global_device_id.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convert_operand_folding",
    srcs = ["convert_operand_folding.cc"],
    hdrs = ["convert_operand_folding.h"],
    deps = [
        ":hlo",
        ":op_expander_pass",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tf_cc_test(
    name = "convert_operand_folding_test",
    srcs = ["convert_operand_folding_test.cc"],
    deps = [
        ":convert_operand_folding",
        ":hlo_matchers",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xla_debug_info_manager",
    srcs = [
        "xla_debug_info_manager.cc",
    ],
    hdrs = [
        "xla_debug_info_manager.h",
    ],
    deps = [
        ":hlo",
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "xla_debug_info_manager_test",
    srcs = ["xla_debug_info_manager_test.cc"],
    deps = [
        ":hlo_proto_cc",
        ":xla_debug_info_manager",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_py_proto_library(
    name = "hlo_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":hlo_proto"],
)

py_library(
    name = "generate_test_hlo_checks",
    srcs = ["generate_test_hlo_checks.py"],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "generate_test_hlo_checks_test",
    srcs = ["generate_test_hlo_checks_test.py"],
    python_version = "PY3",
    # TODO(b/200806426): Test fails in OSS.
    tags = [
        "no_oss",
        "nopip",
    ],
    deps = [
        ":generate_test_hlo_checks",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "mapped_ptr_container_sorter",
    hdrs = ["mapped_ptr_container_sorter.h"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "mapped_ptr_container_sorter_test",
    srcs = ["mapped_ptr_container_sorter_test.cc"],
    deps = [
        ":mapped_ptr_container_sorter",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "rendezvous",
    srcs = ["rendezvous.cc"],
    hdrs = ["rendezvous.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compilation_environments",
    srcs = ["compilation_environments.cc"],
    hdrs = ["compilation_environments.h"],
    deps = [
        "//tensorflow/core:lib",  # fixdeps: keep
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

tf_proto_library(
    name = "test_compilation_environment_proto",
    testonly = 1,
    srcs = ["test_compilation_environment.proto"],
    cc_api_version = 2,
)

tf_cc_test(
    name = "compilation_environments_test",
    srcs = ["compilation_environments_test.cc"],
    deps = [
        ":compilation_environments",
        ":test_compilation_environment_proto_cc",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "layout_normalization",
    srcs = ["layout_normalization.cc"],
    hdrs = ["layout_normalization.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":shape_inference",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "layout_normalization_test",
    srcs = [
        "layout_normalization_test.cc",
    ],
    deps = [
        ":layout_normalization",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)
