load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@local_tsl//tsl/platform:build_config_root.bzl", "if_llvm_x86_available")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir/math:friends"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MathTransforms",
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
    srcs = [
        "math_approximation.cc",
        "math_optimization.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    defines = if_llvm_x86_available(["TF_LLVM_X86_AVAILABLE=1"]),
    deps = [
        ":passes_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ] + if_llvm_x86_available([
        "@llvm-project//mlir:X86VectorDialect",
    ]),
)
