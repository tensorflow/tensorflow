load("//third_party/mlir:tblgen.bzl", "gentbl")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_wrapper_py", "tf_native_cc_binary")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//tensorflow/compiler/...",
        "//tensorflow/lite/experimental/tf_runtime/...",
        "//tensorflow/python/...",
    ],
)

exports_files([
    "ir/tf_generated_ops.td",
    "ir/tf_op_base.td",
    "ir/tf_op_interfaces.td",
    "ir/tf_ops.td",
])

filegroup(
    name = "tensorflow_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffects.td",
    ],
)

gentbl(
    name = "tensorflow_op_interfaces_inc_gen",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "ir/tf_op_interfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ir/tf_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_op_interfaces.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_ops.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_ops.md",
        ),
        (
            "-gen-struct-attr-decls",
            "ir/tf_structs.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "ir/tf_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tf_saved_model_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_saved_model.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_saved_model.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_saved_model.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_saved_model_ops.td",
    td_srcs = [
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
    ],
)

gentbl(
    name = "tensorflow_executor_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_executor.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_executor.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_executor.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    td_srcs = [
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
    ],
)

gentbl(
    name = "tensorflow_device_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_device.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_device.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_device.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_device_ops.td",
    td_srcs = [
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
    ],
)

gentbl(
    name = "tensorflow_canonicalize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "hlo_legalize_tf_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_hlo.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_hlo_patterns.td",
    td_srcs = [
        "//tensorflow/compiler/mlir/xla:hlo_ops_td_files",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tf_legalize_hlo",
    srcs = [
        "transforms/generated_legalize_hlo.inc",
        "transforms/legalize_hlo.cc",
    ],
    deps = [
        ":lower_tf_lib",
        ":tensorflow",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_types",
    srcs = [
        "ir/tf_types.cc",
    ],
    hdrs = [
        "ir/tf_types.def",
        "ir/tf_types.h",
    ],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow",
    srcs = [
        "ir/control_flow_ops.cc",
        "ir/tf_device.cc",
        "ir/tf_executor.cc",
        "ir/tf_executor.cc.inc",
        "ir/tf_executor.h.inc",
        "ir/tf_op_interfaces.cc.inc",
        "ir/tf_op_interfaces.h.inc",
        "ir/tf_ops.cc",
        "ir/tf_ops.cc.inc",
        "ir/tf_ops.h.inc",
        "ir/tf_saved_model.cc",
        "ir/tf_structs.cc",
        "ir/tf_verifiers.cc",
    ],
    hdrs = [
        "ir/control_flow_ops.h",
        "ir/tf_device.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_saved_model.h",
        "ir/tf_structs.h",
        "ir/tf_traits.h",
        "ir/tf_verifiers.h",
        "transforms/bridge.h",
        "transforms/einsum.h",
        "transforms/passes.h",
        "transforms/unroll_batch_matmul.h",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.h",
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":error_util",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_device_ops_inc_gen",
        ":tensorflow_executor_inc_gen",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_ops_inc_gen",
        ":tensorflow_types",
        ":tf_saved_model_inc_gen",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CallOpInterfacesIncGen",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

gentbl(
    name = "decompose_resource_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_decompose_resource_ops.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/decompose_resource_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "decompose_resource_ops",
    srcs = [
        "transforms/decompose_resource_ops.cc",
    ],
    hdrs = [
        "transforms/decompose_resource_ops.h",
    ],
    deps = [
        ":decompose_resource_ops_inc_gen",
        ":tensorflow",
        ":tensorflow_types",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "unroll_batch_matmul_pass",
    srcs = [
        "transforms/unroll_batch_matmul.cc",
    ],
    hdrs = [
        "transforms/unroll_batch_matmul.h",
    ],
    deps = [
        ":tensorflow",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_passes",
    srcs = [
        "transforms/annotate_parameter_replication.cc",
        "transforms/batchmatmul_to_einsum.cc",
        "transforms/bridge.cc",
        "transforms/bridge_pass.cc",
        "transforms/cluster_formation.cc",
        "transforms/cluster_outlining.cc",
        "transforms/collection_ops_util.cc",
        "transforms/decompose_resource_ops_pass.cc",
        "transforms/einsum.cc",
        "transforms/executor_island_coarsening.cc",
        "transforms/executor_tpuv1_inline_tpu_island.cc",
        "transforms/executor_tpuv1_island_coarsening.cc",
        "transforms/executor_tpuv1_outline_tpu_island.cc",
        "transforms/fold_switch.cc",
        "transforms/freeze_global_tensors.cc",
        "transforms/functional_control_flow_to_cfg.cc",
        "transforms/generated_canonicalize.inc",
        "transforms/generated_optimize.inc",
        "transforms/gpu_fusion.cc",
        "transforms/graph_pruning.cc",
        "transforms/launch_to_device_attribute.cc",
        "transforms/layout_optimization.cc",
        "transforms/mark_function_visibility.cc",
        "transforms/materialize_mlir_passthrough_op.cc",
        "transforms/optimize.cc",
        "transforms/optimize_global_tensors.cc",
        "transforms/parallel_execute_to_islands.cc",
        "transforms/promote_resources_to_args.cc",
        "transforms/raise_control_flow.cc",
        "transforms/replicate_invariant_op_hoisting.cc",
        "transforms/replicate_to_island.cc",
        "transforms/resource_device_inference.cc",
        "transforms/resource_op_lifting.cc",
        "transforms/shape_inference.cc",
        "transforms/shape_inference_pass.cc",
        "transforms/sink_constant.cc",
        "transforms/stack_ops_decomposition.cc",
        "transforms/tensor_array_ops_decomposition.cc",
        "transforms/tensor_list_ops_decomposition.cc",
        "transforms/test_side_effect_analysis.cc",
        "transforms/tf_device_assignment.cc",
        "transforms/tpu_cluster_formation.cc",
        "transforms/tpu_dynamic_layout_pass.cc",
        "transforms/tpu_dynamic_padding_mapper.cc",
        "transforms/tpu_extract_outside_compilation.cc",
        "transforms/tpu_merge_variables_with_execute.cc",
        "transforms/tpu_rewrite_pass.cc",
        "transforms/tpu_sharding_identification_pass.cc",
        "transforms/tpu_variable_runtime_reformatting.cc",
        "translate/breakup-islands.cc",
        "translate/control_to_executor_dialect.cc",
        "translate/executor_to_control_dialect.cc",
        "translate/tf_functional_to_executor.cc",
    ],
    hdrs = [
        "transforms/batchmatmul_to_einsum.h",
        "transforms/bridge.h",
        "transforms/collection_ops_util.h",
        "transforms/einsum.h",
        "transforms/passes.h",
        "transforms/shape_inference.h",
    ],
    includes = ["include"],
    deps = [
        ":bridge_logger",
        ":convert_tensor",
        ":convert_type",
        ":decompose_resource_ops",
        ":decompose_resource_ops_inc_gen",
        ":device_util",
        ":error_util",
        ":export_tf_dialect_op",
        ":mangling_util",
        ":side_effect_analysis",
        ":tensorflow",
        ":tensorflow_optimize_inc_gen",
        ":tensorflow_types",
        ":tpu_rewrite_device_util",
        ":translate_utils",
        ":unroll_batch_matmul_pass",
        ":xla_sharding_util",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_test_passes",
    srcs = [
        "transforms/lower_tf_pass.cc",
    ],
    deps = [
        ":lower_tf_lib",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimization_pass",
    srcs = ["transforms/graph_optimization_pass.cc"],
    hdrs = ["transforms/graph_optimization_pass.h"],
    deps = [
        ":error_util",
        ":tensorflow_passes",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimization_pass_registration",
    srcs = ["transforms/graph_optimization_pass_registration.cc"],
    deps = [
        ":graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
    ],
    alwayslink = 1,
)

# Library with TensorFlow dialect static initialization.
cc_library(
    name = "tensorflow_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    deps = [
        ":tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopOpsTransforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert_graphdef",
    srcs = [
        "translate/export_graphdef.cc",
        "translate/import_model.cc",
    ],
    hdrs = [
        "translate/export_graphdef.h",
        "translate/import_model.h",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":error_util",
        ":export_tf_dialect_op",
        ":export_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":tensorflow_passes",
        ":tensorflow_types",
        ":translate_utils",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "parse_text_proto",
    srcs = ["utils/parse_text_proto.cc"],
    hdrs = ["utils/parse_text_proto.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:casts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "import_utils",
    srcs = ["utils/import_utils.cc"],
    hdrs = ["utils/import_utils.h"],
    deps = [
        ":error_util",
        ":parse_text_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "export_utils",
    srcs = [
        "utils/export_utils.cc",
    ],
    hdrs = [
        "utils/export_utils.h",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":mangling_util",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_utils",
    srcs = [
        "utils/translate_utils.cc",
    ],
    hdrs = [
        "utils/translate_utils.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "export_tf_dialect_op",
    srcs = [
        "translate/derived_attr_populator.inc",
        "translate/export_tf_dialect_op.cc",
    ],
    hdrs = [
        "translate/export_tf_dialect_op.h",
    ],
    deps = [
        ":convert_type",
        ":export_utils",
        ":tensorflow",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate_tf_dialect_op",
    srcs = ["translate/translate_tf_dialect_op.cc"],
    deps = [
        ":export_tf_dialect_op",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["translate/mlir_roundtrip_pass.cc"],
    hdrs = ["translate/mlir_roundtrip_pass.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass_registration",
    srcs = ["translate/mlir_roundtrip_pass_registration.cc"],
    deps = [
        ":mlir_roundtrip_pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_import_pass_registration",
    srcs = ["translate/mlir_import_pass_registration.cc"],
    deps = [
        ":mlir_roundtrip_pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["translate/mlir_roundtrip_flags.cc"],
    hdrs = ["translate/mlir_roundtrip_flags.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "convert_type",
    srcs = ["utils/convert_type.cc"],
    hdrs = ["utils/convert_type.h"],
    deps = [
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "convert_type_test",
    size = "small",
    srcs = ["utils/convert_type_test.cc"],
    deps = [
        ":convert_type",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = ["utils/convert_tensor.cc"],
    hdrs = ["utils/convert_tensor.h"],
    deps = [
        ":convert_type",
        ":mangling_util",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "convert_tensor_test",
    size = "small",
    srcs = ["utils/convert_tensor_test.cc"],
    deps = [
        ":convert_tensor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mangling_util",
    srcs = ["utils/mangling_util.cc"],
    hdrs = ["utils/mangling_util.h"],
    deps = [
        ":parse_text_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "decode_constant_pass",
    srcs = [
        "transforms/decode_constant.cc",
    ],
    hdrs = [
        "transforms/decode_constant.h",
    ],
    deps = [
        ":convert_tensor",
        ":tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_passes",
    srcs = [
        "transforms/constant_fold.cc",
        "transforms/dialect_hooks.cc",
    ],
    hdrs = [
        "transforms/constant_fold.h",
    ],
    deps = [
        ":convert_tensor",
        ":decode_constant_pass",
        ":eval_util",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        ":tensorflow_dialect_registration",
        ":tf_dialect_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["transforms/tf_graph_optimization_pass.cc"],
    hdrs = ["transforms/tf_graph_optimization_pass.h"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "eval_util",
    srcs = ["utils/eval_util.cc"],
    hdrs = ["utils/eval_util.h"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":mangling_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_lib",
    srcs = ["translate/tf_mlir_translate.cc"],
    hdrs = ["translate/tf_mlir_translate.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":import_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "translate/tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "translate/tf_mlir_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "translate/tf_mlir_translate_registration.cc",
    ],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

tf_native_cc_binary(
    name = "derived_attr_populator_gen",
    srcs = [
        "translate/derived_attr_populator_gen.cc",
    ],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
        "@llvm-project//mlir:TableGen",
    ],
)

genrule(
    name = "derived_attr_populator_inc",
    srcs = [
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffects.td",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_ops.td",
    ],
    outs = [
        "translate/derived_attr_populator.inc",
    ],
    cmd = ("$(location :derived_attr_populator_gen) " +
           "-I external/llvm-project/mlir/include " +
           "-I external/org_tensorflow " +
           "$(location //tensorflow/compiler/mlir/tensorflow:ir/tf_ops.td) " + " -o $@"),
    tools = [":derived_attr_populator_gen"],
)

filegroup(
    name = "tensorflow_optimize_td_files",
    srcs = [
        "transforms/optimize.td",
    ],
)

gentbl(
    name = "tensorflow_optimize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

COMPILE_MLIR_UTIL_DEPS = [
    ":bridge_logger",
    ":convert_graphdef",
    ":convert_type",
    ":dump_mlir_util",
    ":error_util",
    ":mlir_roundtrip_flags",
    ":tensorflow",
    ":tensorflow_dialect_registration",
    ":tensorflow_passes",
    ":translate_utils",
    "@com_google_absl//absl/types:optional",
    "@llvm-project//llvm:support",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Parser",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:Shape",
    "@llvm-project//mlir:StandardOps",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:Transforms",
    "//tensorflow/compiler/mlir/xla:hlo",
    "//tensorflow/compiler/mlir/xla:mlir_hlo_to_hlo",
    "//tensorflow/compiler/mlir/xla:type_to_shape",
    "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
    "//tensorflow/compiler/mlir/xla:xla_legalize_tf_with_tf2xla",
    "//tensorflow/compiler/tf2xla:common",
    "//tensorflow/compiler/tf2xla:xla_compiler",
    "//tensorflow/core:framework",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/platform:logging",
    "//tensorflow/stream_executor/lib",
    "//tensorflow/compiler/xla:xla_data_proto_cc",
    "//tensorflow/compiler/xla/service:hlo",
]

# Prefer to link 'compile_mlir_util' library that also links necessary
# TensorFlow passes to the pipeline. This library without tf passes is useful
# if the constant folding is not required on the TensorFlow dialect. For
# example, this is used in XLA ondemand compilation which compiles a single op
# at a time and doesn't require constant folding. Doing so helps avoid a
# circular dependency between c_api and tf passes.
# TODO(hinsu): Split out the constant folding hook and only exclude that in
# this target.
cc_library(
    name = "compile_mlir_util_no_tf_dialect_passes",
    srcs = ["utils/compile_mlir_util.cc"],
    hdrs = ["utils/compile_mlir_util.h"],
    deps = COMPILE_MLIR_UTIL_DEPS,
)

cc_library(
    name = "compile_mlir_util",
    hdrs = ["utils/compile_mlir_util.h"],
    deps = COMPILE_MLIR_UTIL_DEPS + [
        "compile_mlir_util_no_tf_dialect_passes",
        ":tf_dialect_passes",
    ],
)

tf_cc_test(
    name = "compile_mlir_util_test",
    size = "small",
    srcs = ["utils/compile_mlir_util_test.cc"],
    deps = [
        ":compile_mlir_util",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "mlir_passthrough_op",
    srcs = ["ops/mlir_passthrough_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_local_var_op",
    srcs = ["ops/mlir_local_var_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_mlir_passthrough_op_py",
    out = "gen_mlir_passthrough_op.py",
    deps = [":mlir_passthrough_op"],
)

# Library to get rewrite patterns lowering within TensorFlow.
#
# This is a separate library so that external passes can link only this library
# without linking any of the other tensorflow passes.
gentbl(
    name = "lower_tf_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_lower_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/lower_tf.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "lower_tf_lib",
    srcs = [
        "transforms/lower_tf.cc",
    ],
    hdrs = [
        "transforms/lower_tf.h",
    ],
    deps = [
        ":lower_tf_inc_gen",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_rewrite_device_util",
    srcs = ["utils/tpu_rewrite_device_util.cc"],
    hdrs = ["utils/tpu_rewrite_device_util.h"],
    deps = [
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
    ],
)

tf_cc_test(
    name = "tpu_rewrite_device_util_test",
    size = "small",
    srcs = ["utils/tpu_rewrite_device_util_test.cc"],
    deps = [
        ":tpu_rewrite_device_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "device_util",
    srcs = ["utils/device_util.cc"],
    hdrs = ["utils/device_util.h"],
    deps = [
        ":tensorflow",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "device_util_test",
    size = "small",
    srcs = ["utils/device_util_test.cc"],
    deps = [
        ":device_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "dump_mlir_util",
    srcs = ["utils/dump_mlir_util.cc"],
    hdrs = ["utils/dump_mlir_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "dump_mlir_util_test",
    size = "small",
    srcs = ["utils/dump_mlir_util_test.cc"],
    deps = [
        ":dump_mlir_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dump_graph",
    srcs = ["utils/dump_graph.cc"],
    hdrs = ["utils/dump_graph.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":tensorflow",
        ":tensorflow_dialect_registration",
        ":tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "dump_graph_test",
    size = "small",
    srcs = ["utils/dump_graph_test.cc"],
    deps = [
        ":dump_graph",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "bridge_logger",
    srcs = ["utils/bridge_logger.cc"],
    hdrs = ["utils/bridge_logger.h"],
    deps = [
        ":dump_mlir_util",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "side_effect_analysis",
    srcs = ["analysis/side_effect_analysis.cc"],
    hdrs = ["analysis/side_effect_analysis.h"],
    deps = [
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_sharding_util",
    srcs = [
        "utils/xla_sharding_util.cc",
    ],
    hdrs = [
        "utils/xla_sharding_util.h",
    ],
    deps = [
        ":tensorflow",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
