load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

td_library(
    name = "xla_gpu_td_files",
    srcs = [
        "xla_gpu_dialect.td",
        "xla_gpu_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    includes = ["include"],
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
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
        (
            ["-gen-op-decls"],
            "xla_gpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xla_gpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_ops.td",
    deps = [":xla_gpu_td_files"],
)

cc_library(
    name = "xla_gpu",
    srcs = [
        "xla_gpu_dialect.cc",
        "xla_gpu_ops.cc",
    ],
    hdrs = [
        "xla_gpu_dialect.h",
        "xla_gpu_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xla_gpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
