load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "legalize_collective_ops.cc",
        "lmhlo_to_cpu_runtime.cc",
        "xla_abi_legalization.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:type_converter",
        "//tensorflow/compiler/xla/mlir/runtime/utils:custom_calls",
        "//tensorflow/compiler/xla/mlir/xla_cpu/ir:xla_cpu",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)
