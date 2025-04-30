# TensorFlow -> TOSA Compiler Bridge.
# See:
#   https://developer.mlplatform.org/w/tosa/
#   https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/TOSA.md

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

# TODO: Tighten visibility once targets are at the right granularity.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/mlir/tosa/...",
    ],
)

filegroup(
    name = "tosa_ops_td_files",
    srcs = [
        "@llvm-project//mlir:TosaDialectTdFiles",
    ],
    compatible_with = get_compatible_with_portable(),
)

gentbl_cc_library(
    name = "tosa_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LegalizeTosa",
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

cc_library(
    name = "passes_header",
    hdrs = [
        "transforms/passes.h",
        "transforms/passes.h.inc",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "legalize_common",
    srcs = [
        "transforms/legalize_common.cc",
        "transforms/legalize_utils.cc",
    ],
    hdrs = [
        "transforms/legalize_common.h",
        "transforms/legalize_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/kernels/internal:common",
        "//tensorflow/compiler/mlir/lite/kernels/internal:quantization_util",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "//tensorflow/lite/kernels/internal:reference_base",
        "@com_google_absl//absl/status",
        "@gemmlowp",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/tsl/framework/fixedpoint",
    ],
)

gentbl_cc_library(
    name = "tosa_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/tf_legalize_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_legalize_patterns.td",
    deps = [
        ":tosa_ops_td_files",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantizationOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_passes",
    srcs = [
        "tf_passes.cc",
        "transforms/dequantize_tfl_softmax.cc",
        "transforms/fuse_bias_tf.cc",
        "transforms/legalize_tf.cc",
        "transforms/tf_legalize_patterns.inc",
    ],
    hdrs = [
        "tf_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tosa_legalize_tf_inc_gen",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "tosa_legalize_tfl_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/tfl_legalize_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tfl_legalize_patterns.td",
    deps = [
        ":tosa_ops_td_files",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tfl_passes",
    srcs = [
        "tfl_passes.cc",
        "transforms/convert_metadata.cc",
        "transforms/convert_tfl_uint8.cc",
        "transforms/legalize_tfl.cc",
        "transforms/legalize_tfl_stateful.cc",
        "transforms/lower_complex_types.cc",
        "transforms/retain_call_once_funcs.cc",
        "transforms/strip_metadata.cc",
        "transforms/strip_quant_types.cc",
        "transforms/tfl_legalize_patterns.inc",
        "transforms/verify_fully_converted.cc",
    ],
    hdrs = [
        "tfl_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tosa_legalize_tfl_inc_gen",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tf_tfl_passes",
    srcs = [
        "tf_tfl_passes.cc",
        "transforms/legalize_tf_tfl.cc",
    ],
    hdrs = [
        "tf_tfl_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tf_passes",
        ":tfl_passes",
        "//tensorflow/compiler/mlir/lite:lift_tflite_flex_ops",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "tf-tosa-opt",
    testonly = True,
    srcs = ["tf_tosa_opt.cc"],
    tags = ["tf_tosa"],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",  # buildcleaner:keep
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//tensorflow/compiler/mlir/tensorflow:mlprogram_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",  # buildcleaner:keep
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:runtime_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms/sparsecore:sparsecore_passes",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "//tensorflow/compiler/mlir/tf2xla/internal/passes:mlir_to_graph_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//tensorflow/compiler/mlir/tosa:tf_passes",
        "//tensorflow/compiler/mlir/tosa:tf_tfl_passes",
        "//tensorflow/compiler/mlir/tosa:tfl_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/ir:xla_framework",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//tensorflow:__subpackages__"],
)
