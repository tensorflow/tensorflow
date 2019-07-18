load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_native_cc_binary")
load(
    "@local_config_mlir//:tblgen.bzl",
    "gentbl",
)

package(
    default_visibility = [
        # TODO(jpienaar): Make the visibility more restrictive.
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["@local_config_mlir//:subpackages"],
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
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:QuantizationOpsTdFiles",
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
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
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
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
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
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/tensorlist_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
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
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/legalize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
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
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
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
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    td_srcs = [
        ":tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
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
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
    ],
)

cc_library(
    name = "tensorflow_lite",
    srcs = [
        "ir/tfl_ops.cc",
        "ir/tfl_ops.cc.inc",
        "ir/tfl_ops.h.inc",
        "utils/attribute_utils.cc",
    ],
    hdrs = [
        "ir/tfl_ops.h",
        "ir/tfl_traits.h",
        "transforms/passes.h",
        "utils/attribute_utils.h",
        "utils/quantization_utils.h",
    ],
    deps = [
        ":tensorflow_lite_ops_inc_gen",
        ":validators",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/lite/schema:schema_fbs",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TypeUtilities",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_lite_quantization_utils",
    srcs = [
        "utils/generated_op_quant_spec_getters.inc",
        "utils/quantization_driver.cc",
        "utils/quantization_utils.cc",
    ],
    hdrs = [
        "utils/quantization_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/generated_lower_static_tensor_list.inc",
        "transforms/generated_prepare_tf.inc",
        "transforms/legalize_tf.cc",
        "transforms/lower_static_tensor_list.cc",
        "transforms/prepare_tf.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_lite",
        ":tensorflow_lite_quantization_utils",
        ":validators",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TypeUtilities",
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
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_lite_quantize",
    srcs = [
        "transforms/generated_quantize.inc",
        "transforms/post_quantize.cc",
        "transforms/prepare_quantize.cc",
        "transforms/quantize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_lite",
        ":tensorflow_lite_quantization_utils",
        ":validators",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "op_quant_spec_getters_gen",
    srcs = [
        "tools/op_quant_spec_getters_gen.cc",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:TableGen",
    ],
)

genrule(
    name = "op_quant_spec_getters_inc",
    srcs = [
        "@local_config_mlir//:include/mlir/Dialect/QuantOps/QuantPredicates.td",
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        ":ir/tfl_ops.td",
    ],
    outs = [
        "utils/generated_op_quant_spec_getters.inc",
    ],
    cmd = ("$(location :op_quant_spec_getters_gen) " +
           "-I external/local_config_mlir/include " +
           "$(location //tensorflow/compiler/mlir/lite:ir/tfl_ops.td) " + " -o $@"),
    tools = [":op_quant_spec_getters_gen"],
)

# Library with tensorflow Lite dialect static initialization.
cc_library(
    name = "tensorflow_lite_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":tensorflow_lite",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "operator-writer-gen",
    srcs = [
        "operator_writer_gen.cc",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:TableGen",
    ],
)

genrule(
    name = "operator_writer_inc",
    srcs = [
        "@local_config_mlir//:include/mlir/Dialect/QuantOps/QuantPredicates.td",
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        ":ir/tfl_ops.td",
    ],
    outs = [
        "operator_writers.inc",
    ],
    cmd = ("$(location :operator-writer-gen) " +
           "-I external/local_config_mlir/include " +
           "$(location //tensorflow/compiler/mlir/lite:ir/tfl_ops.td) " + " -o $@"),
    tools = [":operator-writer-gen"],
)

cc_library(
    name = "flatbuffer_tflite_operator_lib",
    srcs = [
        "flatbuffer_operator.cc",
        "operator_writers.inc",
    ],
    hdrs = [
        "flatbuffer_operator.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:TransformUtils",
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
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "flatbuffer_translate_lib",
    srcs = [
        "flatbuffer_import.cc",
        "flatbuffer_translate.cc",
    ],
    hdrs = [
        "flatbuffer_import.h",
        "flatbuffer_translate.h",
    ],
    deps = [
        ":flatbuffer_tflite_operator_lib",
        ":tensorflow_lite",
        ":tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/delegates/flex:whitelisted_flex_ops_lib",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:QuantOpsDialectRegistration",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "flatbuffer_translate",
    deps = [
        ":flatbuffer_translate_lib",
        "@local_config_mlir//:tools/mlir-translate/mlir-translate",
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
        "@llvm//:support",
    ],
    alwayslink = 1,
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
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        ":tf_tfl_translate_cl_options",
        ":tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

tf_cc_binary(
    name = "mlir-tflite-runner",
    srcs = ["mlir_tflite_runner.cc"],
    deps = [
        ":flatbuffer_translate_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform/default/build_config:base",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "tf_to_tfl_flatbuffer",
    srcs = ["tf_to_tfl_flatbuffer.cc"],
    hdrs = [
        "tf_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        ":tensorflow_lite_legalize_tf",
        ":tensorflow_lite_optimize",
        ":tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:QuantOpsDialectRegistration",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
    ],
)
