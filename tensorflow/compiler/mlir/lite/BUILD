# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        # TODO(jpienaar): Make the visibility more restrictive.
        ":friends",
        "//tensorflow/lite/experimental/tf_runtime:__subpackages__",
        "//tensorflow/lite/testing:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/mlir/...",
        "//third_party/auroraml/...",
        "//third_party/iree/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/lite/python/...",
        "//waymo/ml/compiler/mlir/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
        "//research/language_modeling/sentence_explorer/ondevice/...",
    ],
)

td_library(
    name = "tensorflow_lite_ops_td_files",
    srcs = [
        "ir/tfl_op_enums.td",
        "ir/tfl_op_interfaces.td",
        "ir/tfl_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

td_library(
    name = "tensorflow_lite_patterns_td_files",
    srcs = [
        "transforms/legalize_patterns.td",
        "transforms/legalize_variables.td",
        "transforms/optimize_patterns.td",
        "transforms/post_quantize_patterns.td",
        "transforms/prepare_patterns.td",
        "transforms/quantize_patterns.td",
        "transforms/tensorlist_patterns.td",
        "utils/utils.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_optimize_td_files",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowLite",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tfl_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tfl_ops.cc.inc",
        ),
        (
            [
                "-gen-dialect-doc",
                "-dialect=tfl",
            ],
            "g3doc/tfl_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_ops.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_op_interfaces_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "ir/tfl_ops_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ir/tfl_ops_interface.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "ir/tfl_ops_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "ir/tfl_ops_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_op_interfaces.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_op_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "ir/tfl_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "ir/tfl_ops_enums.cc.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "ir/tfl_ops_attrdefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "ir/tfl_ops_attrdefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_op_enums.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_prepare_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_prepare_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_lower_static_tensor_list_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_lower_static_tensor_list.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tensorlist_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_legalize_variables_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_legalize_variables.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_variables.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_optimize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_quantize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_post_quantize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_post_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/post_quantize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

cc_library(
    name = "validators",
    srcs = [
        "utils/validators.cc",
    ],
    hdrs = [
        "utils/validators.h",
    ],
    deps = [
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow_lite",
    srcs = [
        "ir/tfl_ops.cc",
        "ir/tfl_ops.cc.inc",
        "ir/tfl_ops.h.inc",
        "ir/tfl_ops_attrdefs.h.inc",
        "ir/tfl_ops_dialect.cc.inc",
        "ir/tfl_ops_dialect.h.inc",
        "ir/tfl_ops_enums.cc.inc",
        "ir/tfl_ops_enums.h.inc",
        "ir/tfl_ops_interface.cc.inc",
        "ir/tfl_ops_interface.h.inc",
        "runtime_verifiers.inc",
        "utils/attribute_utils.cc",
    ],
    hdrs = [
        "ir/tfl_ops.h",
        "transforms/passes.h",
        "utils/attribute_utils.h",
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    deps = [
        ":cost_estimators",
        ":tensorflow_lite_op_enums_inc_gen",
        ":tensorflow_lite_op_interfaces_inc_gen",
        ":tensorflow_lite_ops_inc_gen",
        ":tensorflow_lite_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "variables_utils",
    srcs = [
        "utils/variables_utils.cc",
    ],
    hdrs = [
        "utils/variables_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
    ],
)

cc_library(
    name = "cost_estimators",
    hdrs = [
        "utils/arithmetic_count_util.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "constant_utils",
    srcs = [
        "utils/constant_utils.cc",
    ],
    hdrs = [
        "utils/constant_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "lstm_utils",
    srcs = [
        "utils/lstm_utils.cc",
    ],
    hdrs = [
        "utils/lstm_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "fake_quant_utils",
    srcs = [
        "utils/fake_quant_utils.cc",
    ],
    hdrs = [
        "utils/fake_quant_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "nms_utils",
    srcs = [
        "utils/nms_utils.cc",
    ],
    hdrs = [
        "utils/nms_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/core:framework",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tftext_utils",
    srcs = [
        "utils/tftext_utils.cc",
    ],
    hdrs = [
        "utils/tftext_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/core:framework",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tftext_utils_test",
    size = "small",
    srcs = ["utils/tftext_utils_test.cc"],
    deps = [
        ":tftext_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "perception_ops_utils",
    srcs = [
        "utils/perception_ops_utils.cc",
    ],
    hdrs = [
        "utils/perception_ops_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/lite/c:common",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stateful_ops_utils",
    srcs = [
        "utils/stateful_ops_utils.cc",
    ],
    hdrs = [
        "utils/stateful_ops_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "lstm_utils_test",
    size = "small",
    srcs = ["utils/lstm_utils_test.cc"],
    deps = [
        ":lstm_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

tf_cc_test(
    name = "perception_ops_utils_test",
    size = "small",
    srcs = ["utils/perception_ops_utils_test.cc"],
    deps = [
        ":perception_ops_utils",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf",
    srcs = [
        "transforms/analyze_variables.cc",
        "transforms/dilated_conv.cc",
        "transforms/generated_legalize_tf.inc",
        "transforms/generated_legalize_variables.inc",
        "transforms/generated_lower_static_tensor_list.inc",
        "transforms/generated_prepare_tf.inc",
        "transforms/get_arithmetic_count.cc",
        "transforms/insert_call_once_op.cc",
        "transforms/legalize_hashtables.cc",
        "transforms/legalize_jax_random.cc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_while.cc",
        "transforms/legalize_variables.cc",
        "transforms/lower_static_tensor_list.cc",
        "transforms/optimize_functional_ops.cc",
        "transforms/prepare_composite_functions_tf.cc",
        "transforms/prepare_tf.cc",
        "transforms/raise_custom_ops.cc",
        "transforms/reduce_while_operands.cc",
        "transforms/runtime_verify.cc",
        "transforms/split_merged_operands.cc",
        "transforms/trim_functions_tf.cc",
        "transforms/unfold_large_splat_constant.cc",
        "transforms/while_loop_outline.cc",
    ],
    hdrs = [
        "ir/tfl_ops_interface.h.inc",
        "transforms/dilated_conv.h",
        "transforms/passes.h",
    ],
    deps = [
        ":constant_utils",
        ":cost_estimators",
        ":fake_quant_utils",
        ":lstm_utils",
        ":nms_utils",
        ":perception_ops_utils",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        ":tftext_utils",
        ":validators",
        ":variables_utils",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tensorflow:unroll_batch_matmul_pass",
        "//tensorflow/compiler/mlir/tensorflow:verification_utils",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:tensor_list",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tensorflow_lite_optimize",
    srcs = [
        "transforms/generated_optimize.inc",
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":convert_type",
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        ":validators",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:verification_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tensorflow_lite_quantize",
    srcs = [
        "transforms/decompose_hybrid_quantization.cc",
        "transforms/default_quant_params.cc",
        "transforms/generated_post_quantize.inc",
        "transforms/generated_quantize.inc",
        "transforms/modify_io_nodes.cc",
        "transforms/optimize_op_order.cc",
        "transforms/post_quantize.cc",
        "transforms/prepare_quantize.cc",
        "transforms/prepare_quantize_dynamic_range.cc",
        "transforms/quantize.cc",
        "utils/generated_op_quant_spec_getters.inc",
    ],
    hdrs = [
        "transforms/passes.h",
        "transforms/prepare_quantize_helper.h",
    ],
    deps = [
        "convert_type",
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        ":validators",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/lite/quantization/lite:tfl_to_std",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:operator_property",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tensorflow_lite_d2s",
    srcs = [
        "transforms/dense_to_sparse.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "lift_tflite_flex_ops",
    srcs = [
        "transforms/lift_tflite_flex_ops.cc",
    ],
    hdrs = [
        "transforms/lift_tflite_flex_ops.h",
    ],
    deps = [
        ":tensorflow_lite_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_attr",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

filegroup(
    name = "generated_op_quant_spec_getters",
    srcs = [
        "utils/generated_op_quant_spec_getters.inc",
    ],
)

gentbl_cc_library(
    name = "op_quant_spec_getters_inc",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [([], "utils/generated_op_quant_spec_getters.inc")],
    tblgen = "//tensorflow/compiler/mlir/lite/quantization:op_quant_spec_getters_gen",
    td_file = "ir/tfl_ops.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tflite_op_coverage_spec_inc",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [([], "utils/tflite_op_coverage_spec.inc")],
    tblgen = "//tensorflow/compiler/mlir/lite/quantization:tflite_op_coverage_spec_getters_gen",
    td_file = "ir/tfl_ops.td",
    visibility = ["//learning/brain/mobile/model_optimization/g3doc/autogen:__pkg__"],
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

tf_native_cc_binary(
    name = "converter-gen",
    srcs = [
        "converter_gen.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "converter_inc",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["--gen-operator-converters"],
            "operator_converters.inc",
        ),
        (
            ["--gen-runtime-verifiers"],
            "runtime_verifiers.inc",
        ),
    ],
    tblgen = ":converter-gen",
    td_file = "ir/tfl_ops.td",
    test = 1,
    deps = [
        ":tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
)

cc_library(
    name = "flatbuffer_tflite_operator_lib",
    srcs = [
        "flatbuffer_operator.cc",
        "operator_converters.inc",
    ],
    hdrs = [
        "flatbuffer_operator.h",
    ],
    deps = [
        ":convert_type",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_native_cc_binary(
    name = "flatbuffer_to_string",
    srcs = ["flatbuffer_to_string.cc"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
)

tf_native_cc_binary(
    name = "json_to_flatbuffer",
    srcs = ["json_to_flatbuffer.cc"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_library(
    name = "emit_error_reporter",
    srcs = [
        "emit_error_reporter.cc",
    ],
    hdrs = [
        "emit_error_reporter.h",
    ],
    deps = [
        "//tensorflow/lite/core/api",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "flatbuffer_export",
    srcs = [
        "flatbuffer_export.cc",
    ],
    hdrs = [
        "flatbuffer_export.h",
        "flatbuffer_export_flags.h",
    ],
    deps = [
        ":convert_type",
        ":flatbuffer_tflite_operator_lib",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector_inst",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:status",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/delegates/flex:allowlisted_flex_ops_lib",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/tools/versioning",
        "//tensorflow/lite/tools/versioning:gpu_compatibility",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "flatbuffer_import",
    srcs = [
        "flatbuffer_import.cc",
    ],
    hdrs = [
        "flatbuffer_import.h",
    ],
    deps = [
        ":convert_type",
        ":flatbuffer_tflite_operator_lib",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "convert_type",
    srcs = [
        "utils/convert_type.cc",
    ],
    hdrs = [
        "utils/convert_type.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/lite/schema:schema_fbs",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "flatbuffer_translate_lib",
    hdrs = [
        "flatbuffer_export.h",
        "flatbuffer_export_flags.h",
        "flatbuffer_import.h",
        "utils/convert_type.h",
    ],
    deps = [
        ":flatbuffer_export",
        ":flatbuffer_import",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "flatbuffer_translate_registeration",
    srcs = [
        "flatbuffer_translate.cc",
    ],
    deps = [
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirTranslateMain",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "flatbuffer_translate",
    deps = [
        ":flatbuffer_translate_registeration",
    ],
)

cc_library(
    name = "tf_tfl_translate_cl_options",
    srcs = [
        "tf_tfl_translate_cl.cc",
    ],
    hdrs = [
        "tf_tfl_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "common",
    hdrs = [
        "common/tfl_pass_config.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "tf_tfl_translate_main",
    srcs = [
        "tf_tfl_translate.cc",
    ],
)

tf_cc_binary(
    name = "tf_tfl_translate",
    srcs = [
        ":tf_tfl_translate_main",
    ],
    deps = [
        ":common",
        ":flatbuffer_translate_lib",
        ":flatbuffer_translate_registeration",
        ":tensorflow_lite",
        ":tf_tfl_passes",
        ":tf_tfl_translate_cl_options",
        ":tf_to_tfl_flatbuffer",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/xla:xla_mlir_translate",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "mlir-tflite-runner",
    srcs = [
        "mlir_tflite_runner.cc",
    ],
    deps = [
        ":flatbuffer_translate_lib",
        ":flatbuffer_translate_registeration",
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "tf_tfl_passes",
    srcs = ["tf_tfl_passes.cc"],
    hdrs = [
        "tf_tfl_passes.h",
    ],
    deps = [
        ":common",
        ":fake_quant_utils",
        ":tensorflow_lite_d2s",
        ":tensorflow_lite_legalize_tf",
        ":tensorflow_lite_optimize",
        ":tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:expand_hlo_tuples",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "//tensorflow/compiler/mlir/lite/quantization/tensorflow:tf_quantization_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tf_to_tfl_flatbuffer",
    srcs = ["tf_to_tfl_flatbuffer.cc"],
    hdrs = [
        "tf_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":common",
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        ":tensorflow_lite_legalize_tf",
        ":tensorflow_lite_optimize",
        ":tensorflow_lite_quantize",
        ":tf_tfl_passes",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector_inst",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_freeze_variables",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/tools/optimize:quantize_weights",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "empty_passes",
    hdrs = ["transforms/passes.h"],
    visibility = [
        "//configs/devtools/hawkeye/tflite:__subpackages__",
        "//learning/brain/models/app_benchmarks:__subpackages__",
    ],
    deps = [
        ":tensorflow_lite_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

build_test(
    name = "tensorflow_lite_build_test",
    targets = [
        ":tensorflow_lite",
    ],
)
