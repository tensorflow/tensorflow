load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")
load(
    "//third_party/mlir:tblgen.bzl",
    "gentbl",
)

package(
    default_visibility = [
        # TODO(jpienaar): Make the visibility more restrictive.
        ":friends",
        "//tensorflow/lite/experimental/tf_runtime:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/...",
        "//tensorflow/compiler/mlir/...",
    ],
)

filegroup(
    name = "tensorflow_lite_ops_td_files",
    srcs = [
        "ir/tfl_ops.td",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "tensorflow_lite_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tfl_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tfl_ops.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/tfl_ops.md",
        ),
        (
            "-gen-op-interface-decls",
            "ir/tfl_ops_interface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ir/tfl_ops_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_ops.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_lite_prepare_tf_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_prepare_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_optimize_td_files",
    ],
)

gentbl(
    name = "tensorflow_lite_lower_static_tensor_list_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_lower_static_tensor_list.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tensorlist_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_lite_legalize_tf_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_lite_optimize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_lite_quantize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "tensorflow_lite_post_quantize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_post_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/post_quantize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "tensorflow_lite",
    srcs = [
        "ir/tfl_ops.cc",
        "ir/tfl_ops.cc.inc",
        "ir/tfl_ops.h.inc",
        "ir/tfl_ops_interface.cc.inc",
        "ir/tfl_ops_interface.h.inc",
        "utils/attribute_utils.cc",
    ],
    hdrs = [
        "ir/tfl_ops.h",
        "transforms/passes.h",
        "utils/attribute_utils.h",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_traits.h",
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    deps = [
        ":tensorflow_lite_ops_inc_gen",
        ":validators",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/lite/schema:schema_fbs",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lstm_utils",
    srcs = [
        "utils/lstm_utils.cc",
    ],
    hdrs = [
        "utils/lstm_utils.h",
    ],
    copts = ["-std=c++14"],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
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
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
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
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf",
    srcs = [
        "transforms/extract_ophint.cc",
        "transforms/generated_legalize_tf.inc",
        "transforms/generated_lower_static_tensor_list.inc",
        "transforms/generated_prepare_tf.inc",
        "transforms/legalize_ophint_func_op.cc",
        "transforms/legalize_tf.cc",
        "transforms/lower_static_tensor_list.cc",
        "transforms/optimize_functional_ops.cc",
        "transforms/prepare_composite_functions_tf.cc",
        "transforms/prepare_tf.cc",
        "transforms/split_merged_operands.cc",
        "transforms/trim_functions_tf.cc",
        "transforms/unroll_batch_matmul.cc",
    ],
    hdrs = [
        "transforms/passes.h",
        "transforms/unroll_batch_matmul.h",
    ],
    deps = [
        ":common",
        ":lstm_utils",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        ":validators",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
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
        ":tensorflow_lite",
        ":validators",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_lite_quantize",
    srcs = [
        "transforms/default_quant_params.cc",
        "transforms/generated_post_quantize.inc",
        "transforms/generated_quantize.inc",
        "transforms/load_quantization_recipe.cc",
        "transforms/post_quantize.cc",
        "transforms/prepare_quantize.cc",
        "transforms/quantize.cc",
        "utils/generated_op_quant_spec_getters.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_lite",
        ":validators",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "generated_op_quant_spec_getters",
    srcs = [
        "utils/generated_op_quant_spec_getters.inc",
    ],
)

genrule(
    name = "op_quant_spec_getters_inc",
    srcs = [
        "ir/tfl_ops.td",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
    outs = [
        "utils/generated_op_quant_spec_getters.inc",
    ],
    cmd = ("$(location //tensorflow/compiler/mlir/lite/quantization:op_quant_spec_getters_gen) " +
           "-I external/llvm-project/mlir/include " +
           "-I external/org_tensorflow " +
           "$(location //tensorflow/compiler/mlir/lite:ir/tfl_ops.td) " + " -o $@"),
    tools = ["//tensorflow/compiler/mlir/lite/quantization:op_quant_spec_getters_gen"],
)

# Library with tensorflow Lite dialect static initialization.
cc_library(
    name = "tensorflow_lite_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":tensorflow_lite",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "operator-converter-gen",
    srcs = [
        "operator_converter_gen.cc",
    ],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl(
    name = "operator_converter_inc",
    tbl_outs = [
        (
            "",  # This driver has no options.
            "operator_converters.inc",
        ),
    ],
    tblgen = ":operator-converter-gen",
    td_file = "ir/tfl_ops.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
    test = 1,
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
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:support",
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
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "flatbuffer_translate_lib",
    srcs = [
        "flatbuffer_import.cc",
        "flatbuffer_translate.cc",
        "utils/convert_type.cc",
    ],
    hdrs = [
        "flatbuffer_import.h",
        "flatbuffer_translate.h",
        "flatbuffer_translate_flags.h",
        "utils/convert_type.h",
    ],
    deps = [
        ":flatbuffer_tflite_operator_lib",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        ":tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/delegates/flex:whitelisted_flex_ops_lib",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/versioning:op_version",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:QuantOpsDialectRegistration",
        "@llvm-project//mlir:StandardDialectRegistration",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "flatbuffer_translate",
    deps = [
        ":flatbuffer_translate_lib",
        "@llvm-project//mlir:MlirTranslateMain",
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
        "@llvm-project//llvm:support",
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
        "@llvm-project//llvm:support",
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
    srcs = [":tf_tfl_translate_main"],
    deps = [
        ":common",
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        ":tf_tfl_passes",
        ":tf_tfl_translate_cl_options",
        ":tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_binary(
    name = "mlir-tflite-runner",
    srcs = ["mlir_tflite_runner.cc"],
    deps = [
        ":flatbuffer_translate_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
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
        ":tensorflow_lite_legalize_tf",
        ":tensorflow_lite_optimize",
        ":tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:QuantOpsDialectRegistration",
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
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/tools/optimize:quantize_weights",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:QuantOpsDialectRegistration",
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
        "//tensorflow/compiler/mlir/lite:friends",
        "//tensorflow/lite/experimental/mlir:__subpackages__",
    ],
    deps = [
        "@llvm-project//llvm:support",
    ],
)
