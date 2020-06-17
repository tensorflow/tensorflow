load("//third_party/mlir:tblgen.bzl", "gentbl")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "tfjs_ops_td_files",
    srcs = [
        "ir/tfjs_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "tfjs_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tfjs_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tfjs_ops.cc.inc",
        ),
        (
            "-gen-dialect-decls",
            "ir/tfjs_dialect.h.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tfjs_dialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfjs_ops.td",
    td_srcs = [
        "ir/tfjs_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_js_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":tensorflow_js",
    ],
    alwayslink = 1,
)

gentbl(
    name = "tfjs_optimize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_pattern.td",
    td_srcs = [
        ":tfjs_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tfjs_optimize",
    srcs = [
        "transforms/generated_optimize.inc",
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tensorflow_js",
        ":tensorflow_js_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
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
        "//tensorflow/compiler/mlir/tensorflow:decode_constant_pass",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
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
        ":tensorflow_js_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:export_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
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
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:decode_constant_pass",
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
        "@llvm-project//mlir:AllPassesAndDialects",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
