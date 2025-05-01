load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary")

# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

# TODO(b/249181641): Restructure source codes for TF quantization.
package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo/...",
    ] + internal_visibility_allowlist(),
)

py_strict_binary(
    name = "gen_quantized_function_library",
    srcs = ["gen_quantized_function_library.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

genrule(
    name = "quantized_function_library",
    srcs = [
        "passes/quantized_function_library_uniform_quantized.mlir",
        "passes/quantized_function_library.mlir",
        "passes/quantized_function_library_uniform_quantized_drq.mlir",
        "passes/quantized_function_library_tf_drq.mlir",
        "passes/quantized_function_library_xla_weight_only.mlir",
    ],
    outs = [
        "passes/quantized_function_library.h",
    ],
    cmd = "$(location gen_quantized_function_library) --output_file $(RULEDIR)/passes/quantized_function_library.h --src '$(SRCS)'",
    compatible_with = get_compatible_with_portable(),
    tools = ["gen_quantized_function_library"],
)

cc_library(
    name = "manipulate_model_attr",
    srcs = [
        "passes/manipulate_model_attr.cc",
    ],
    hdrs = [
        "passes/manipulate_model_attr.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "remove_identity_op_pattern",
    srcs = [
        "passes/remove_identity_op_pattern.cc",
    ],
    hdrs = [
        "passes/remove_identity_op_pattern.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

td_library(
    name = "tf_quant_td_files",
    srcs = [
        "passes/tf_post_quantize.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common:quant_td_files",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantizationOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

td_library(
    name = "quant_td_files",
    srcs = [
        "passes/cast_bf16_ops_to_f32.td",
        "passes/lift_quantizable_spots_as_functions.td",
        "passes/lift_quantizable_spots_as_functions_drq.td",
        "passes/optimize.td",
        "passes/post_quantize.td",
        "passes/prepare_lifting.td",
        "passes/prepare_quantize.td",
        "passes/preprocess_op.td",
        "passes/quantize_composite_functions.td",
        "passes/replace_cast_hacks_with_tf_xla_ops.td",
        "passes/tf_quant_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/quantization/common:quant_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

gentbl_cc_library(
    name = "convert_tf_xla_op_to_tf_op_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/convert_tf_xla_op_to_tf_op.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/convert_tf_xla_op_to_tf_op.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "cast_bf16_ops_to_f32_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/cast_bf16_ops_to_f32.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/cast_bf16_ops_to_f32.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "prepare_lifting_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/prepare_lifting.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/prepare_lifting.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_drq_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions_drq.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions_drq.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "prepare_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/prepare_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/prepare_quantize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "quantize_composite_functions_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/quantize_composite_functions.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/quantize_composite_functions.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "tf_quant_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "passes/tf_quant_ops.h.inc": ["-gen-op-decls"],
        "passes/tf_quant_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/tf_quant_ops.td",
    deps = [
        ":quant_td_files",
    ],
)

gentbl_cc_library(
    name = "optimize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/optimize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/optimize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "convert_tpu_model_to_cpu_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/convert_tpu_model_to_cpu.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/convert_tpu_model_to_cpu.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "post_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/post_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/post_quantize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "tf_post_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/tf_post_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/tf_post_quantize.td",
    deps = [":tf_quant_td_files"],
)

gentbl_cc_library(
    name = "preprocess_op_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/preprocess_op.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/preprocess_op.td",
    deps = [":quant_td_files"],
)

cc_library(
    name = "tf_quant_ops",
    srcs = [
        "passes/tf_quant_ops.cc",
        "passes/tf_quant_ops.cc.inc",
        "passes/tf_quant_ops.h.inc",
    ],
    hdrs = ["passes/tf_quant_ops.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_side_effects",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_structs",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_traits",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "replace_cast_hacks_with_tf_xla_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/replace_cast_hacks_with_tf_xla_ops.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/replace_cast_hacks_with_tf_xla_ops.td",
    deps = [":quant_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "passes/add_dump_tensor_op.cc",
        "passes/add_quantization_unit_loc.cc",
        "passes/cast_bf16_ops_to_f32.cc",
        "passes/cast_bf16_ops_to_f32.inc",
        "passes/convert_custom_aggregation_op_to_quant_stats.cc",
        "passes/convert_fake_quant_to_qdq.cc",
        "passes/convert_tf_xla_op_to_tf_op.cc",
        "passes/convert_tf_xla_op_to_tf_op.inc",
        "passes/convert_tpu_model_to_cpu.cc",
        "passes/convert_tpu_model_to_cpu.inc",
        "passes/duplicate_shape_determining_constants.cc",
        "passes/insert_custom_aggregation_ops.cc",
        "passes/insert_main_function.cc",
        "passes/insert_quantized_functions.cc",
        "passes/insert_restore_op.cc",
        "passes/insert_save_op.cc",
        "passes/lift_hashtable_ops_as_args.cc",
        "passes/lift_quantizable_spots_as_functions.cc",
        "passes/lift_quantizable_spots_as_functions.inc",
        "passes/lift_quantizable_spots_as_functions_drq.cc",
        "passes/lift_quantizable_spots_as_functions_drq.inc",
        "passes/mark_functions_noinline.cc",
        "passes/merge_duplicate_resource_ops.cc",
        "passes/merge_initializer_function_ops_to_main.cc",
        "passes/merge_save_function_ops_to_main.cc",
        "passes/optimize.cc",
        "passes/optimize.inc",
        "passes/post_quantize.cc",
        "passes/post_quantize.inc",
        "passes/prepare_lifting.cc",
        "passes/prepare_lifting.inc",
        "passes/prepare_quantize.cc",
        "passes/prepare_quantize.inc",
        "passes/prepare_quantize_drq.cc",
        "passes/preprocess_op.cc",
        "passes/preprocess_op.inc",
        "passes/propagate_quantize_type.cc",
        "passes/quantize.cc",
        "passes/quantize_composite_functions.cc",
        "passes/quantize_composite_functions.inc",
        "passes/quantize_weights.cc",
        "passes/quantized_function_library.h",
        "passes/remove_var_init_by_const.cc",
        "passes/replace_cast_hacks_with_tf_xla_ops.cc",
        "passes/replace_cast_hacks_with_tf_xla_ops.inc",
        "passes/unfreeze_constants.cc",
    ],
    hdrs = [
        "passes/constants.h",
        "passes/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":cast_bf16_ops_to_f32_inc_gen",
        ":convert_tf_xla_op_to_tf_op_inc_gen",
        ":convert_tpu_model_to_cpu_inc_gen",
        ":lift_quantizable_spots_as_functions_drq_inc_gen",
        ":lift_quantizable_spots_as_functions_inc_gen",
        ":manipulate_model_attr",
        ":optimize_inc_gen",
        ":post_quantize_inc_gen",
        ":prepare_lifting_inc_gen",
        ":prepare_quantize_inc_gen",
        ":preprocess_op_gen",
        ":quantization_options_proto_cc",
        ":quantize_composite_functions_inc_gen",
        ":remove_identity_op_pattern",
        ":replace_cast_hacks_with_tf_xla_ops_inc_gen",
        ":tf_quant_ops",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common:func",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:const_op_size",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:constant_fold",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:quantization_unit_loc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/ops:tf_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow/ops:tf_quantize_op",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:fake_quant_utils",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:tf_to_uniform_attribute_utils",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:tf_to_xla_attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:xla_call_module_attrs",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//tensorflow/compiler/mlir/utils:name_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/importexport:convert_tensor",
        "//tensorflow/core/ir/importexport:mangling",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@local_tsl//tsl/platform:str_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
        "@stablehlo//:chlo_ops",
    ],
    # Alwayslink is required for registering the MLIR passes.
    # TODO(b/255530126): Split the pass registration from the definitions to avoid binary size bloat.
    alwayslink = True,
)

cc_library(
    name = "tf_passes",
    srcs = [
        "passes/lift_quantizable_spots_as_functions.inc",
        "passes/prepare_lifting.inc",
        "passes/prepare_quantize.inc",
        "passes/tf_add_quantization_unit_loc.cc",
        "passes/tf_convert_fake_quant_to_qdq.cc",
        "passes/tf_lift_quantizable_spots_as_functions.cc",
        "passes/tf_post_quantize.cc",
        "passes/tf_post_quantize.inc",
        "passes/tf_prepare_lifting.cc",
        "passes/tf_prepare_quantize.cc",
        "passes/tf_prepare_quantize_drq.cc",
        "passes/tf_preprocess_op.cc",
        "passes/tf_quantize.cc",
    ],
    hdrs = [
        "passes/tf_passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":lift_quantizable_spots_as_functions_inc_gen",
        ":prepare_lifting_inc_gen",
        ":prepare_quantize_inc_gen",
        ":preprocess_op_gen",
        ":quantization_options_proto_cc",
        ":remove_identity_op_pattern",
        ":tf_post_quantize_inc_gen",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/common:tf_attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/tf_quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/tf_quantization_lib:tf_quantization_config",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:constant_fold",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:quantization_unit_loc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/ops:temp_tf_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow/ops:tf_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:temp_fake_quant_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    # Alwayslink is required for registering the MLIR passes.
    # TODO(b/255530126): Split the pass registration from the definitions to avoid binary size bloat.
    alwayslink = True,
)

cc_library(
    name = "quantize_preprocess",
    srcs = [
        "quantize_preprocess.cc",
    ],
    hdrs = [
        "quantize_preprocess.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:pass_pipeline",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/stablehlo:fold_broadcast_pass",
        "//tensorflow/compiler/mlir/stablehlo:fuse_convolution_pass",
        "//tensorflow/compiler/mlir/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "//tensorflow/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "//tensorflow/compiler/mlir/stablehlo:tf_stablehlo",
        "//tensorflow/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_freeze_variables",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:all_passes",
    ],
)

cc_library(
    name = "quantize_passes",
    srcs = ["quantize_passes.cc"],
    hdrs = ["quantize_passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes",
        ":quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:pass_pipeline",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
    ],
)

# OSS only: This target is header-only. Link `quantization_options_proto_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link
# `quantization_options_proto_impl` directly unless the target does not link
# `libtensorflow_framework.so`.
tf_proto_library(
    name = "quantization_options_proto",
    srcs = [
        "quantization_options.proto",
    ],
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core:protos_all",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_options_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":quantization_options_proto"],
# )
# copybara:uncomment_end

# OSS only: This target is header-only. Link `exported_model_proto_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link
# `exported_model_proto_impl` directly unless the target does not link
# `libtensorflow_framework.so`.
tf_proto_library(
    name = "exported_model_proto",
    srcs = ["exported_model.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = [
        ":internal_visibility_allowlist_package",
        # To be visible from `lib_internal_impl`.
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "exported_model_py_pb2",
#     deps = [":exported_model_proto"],
# )
# copybara:uncomment_end

tf_cc_binary(
    name = "tf-quant-opt",
    srcs = ["passes/tf_quant_opt.cc"],
    deps = [
        ":passes",
        ":tf_passes",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)
