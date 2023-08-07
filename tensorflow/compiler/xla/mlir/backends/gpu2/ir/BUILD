load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "xla_gpu_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "xla_gpu_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "xla_gpu_dialect.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "xla_gpu_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "xla_gpu_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_dialect.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

cc_library(
    name = "xla_gpu",
    srcs = ["xla_gpu_dialect.cc"],
    hdrs = ["xla_gpu_dialect.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xla_gpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
