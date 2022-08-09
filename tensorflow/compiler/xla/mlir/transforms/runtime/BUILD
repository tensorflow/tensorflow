load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=RuntimeTransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = ["convert_to_entrypoint.cc"],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "type_converter",
    srcs = ["type_converter.cc"],
    hdrs = ["type_converter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
        "//tensorflow/compiler/xla/runtime:types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
    ],
)
