load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "td_files",
    srcs = [
        "xla_framework_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_framework_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "xla_framework.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xla_framework.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "xla_framework_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "xla_framework_dialect.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "xla_framework_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "xla_framework_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_framework_ops.td",
    deps = [":td_files"],
)

cc_library(
    name = "xla_framework",
    srcs = [
        "xla_framework.cc",
        "xla_framework.cc.inc",
        "xla_framework.h.inc",
    ],
    hdrs = ["xla_framework.h"],
    deps = [
        ":xla_framework_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)
