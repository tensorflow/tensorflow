load("//third_party/mlir:tblgen.bzl", "gentbl")

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
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffects.td",
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
        "@llvm-project//llvm:support",
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
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
