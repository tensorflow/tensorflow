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
                "-name=MemrefTransforms",
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
    srcs = ["aligned_allocations.cc"],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
