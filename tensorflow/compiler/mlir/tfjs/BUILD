# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "tfjs_ops_td_files",
    srcs = [
        "ir/tfjs_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tfjs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tfjs_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tfjs_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "ir/tfjs_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "ir/tfjs_dialect.cc.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "g3doc/tfjs_dialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfjs_ops.td",
    deps = [
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

gentbl_cc_library(
    name = "tfjs_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TFJS",
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
    name = "tensorflow_js",
    srcs = [
        "ir/tfjs_dialect.h.inc",
        "ir/tfjs_ops.cc",
        "ir/tfjs_ops.cc.inc",
        "ir/tfjs_ops.h.inc",
    ],
    hdrs = [
        "ir/tfjs_ops.h",
    ],
    deps = [
        ":tfjs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "tfjs_optimize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_pattern.td",
    deps = [
        ":tfjs_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tfjs_optimize",
    srcs = [
        "transforms/generated_optimize.inc",
        "transforms/optimize.cc",
        "transforms/passes.h.inc",
        "transforms/passes_detail.h",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_js",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_js_passes",
    srcs = ["tf_tfjs_passes.cc"],
    hdrs = [
        "tf_tfjs_passes.h",
    ],
    deps = [
        ":tfjs_optimize",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "json_translate_lib",
    srcs = [
        "translate/json_translate.cc",
    ],
    hdrs = [
        "translate/json_translate.h",
    ],
    deps = [
        ":tensorflow_js",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:export_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_to_tfjs_json",
    srcs = ["translate/tf_to_tfjs_json.cc"],
    hdrs = [
        "translate/tf_to_tfjs_json.h",
    ],
    deps = [
        ":json_translate_lib",
        ":tfjs_optimize",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "json_translate",
    deps = [
        ":json_translate_lib",
        "@llvm-project//mlir:MlirTranslateMain",
    ],
)

filegroup(
    name = "tf_tfjs_translate_main",
    srcs = [
        "translate/tf_tfjs_translate.cc",
    ],
)

tf_cc_binary(
    name = "tf_tfjs_translate",
    srcs = [":tf_tfjs_translate_main"],
    deps = [
        ":json_translate_lib",
        ":tensorflow_js_passes",
        ":tf_to_tfjs_json",
        ":tfjs_optimize",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_binary(
    name = "tfjs-opt",
    srcs = [
        "tfjs_opt.cc",
    ],
    deps = [
        ":tensorflow_js",
        ":tensorflow_js_passes",
        ":tfjs_optimize",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite:lift_tflite_flex_ops",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
