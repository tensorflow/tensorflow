load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CpuTransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "lmhlo_to_cpu_runtime.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/utils/runtime:custom_calls",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)
