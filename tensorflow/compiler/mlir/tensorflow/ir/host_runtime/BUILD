load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "tensorflow_tfrt_ops_td_files",
    srcs = [
        "tfrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_tfrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tfrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tfrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tfrt_ops.td",
    deps = [
        ":tensorflow_tfrt_ops_td_files",
    ],
)

cc_library(
    name = "tensorflow_tfrt_ops",
    srcs = [
        "tfrt_ops.cc",
    ],
    hdrs = [
        "tfrt_ops.h",
    ],
    textual_hdrs = [
        "tfrt_ops.h.inc",
    ],
    deps = [
        ":tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:side_effect_analysis_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_traits",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/transforms:rewrite_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_canonicalize_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:resource_handle",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
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
