load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

td_library(
    name = "xla_gpu_td_files",
    srcs = glob(["*.td"]),
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        "//xla/codegen/emitters/ir:xla_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_gpu_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_gpu_dialect.h.inc": ["-gen-dialect-decls"],
        "xla_gpu_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_dialect.td",
    deps = [":xla_gpu_td_files"],
)

gentbl_cc_library(
    name = "xla_gpu_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_gpu_ops.h.inc": ["-gen-op-decls"],
        "xla_gpu_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_ops.td",
    deps = [":xla_gpu_td_files"],
)

gentbl_cc_library(
    name = "xla_gpu_attrs_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_gpu_enums.h.inc": ["-gen-enum-decls"],
        "xla_gpu_enums.cc.inc": ["-gen-enum-defs"],
        "xla_gpu_attrs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=xla_gpu",
        ],
        "xla_gpu_attrs.cc.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=xla_gpu",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_attrs.td",
    deps = [":xla_gpu_td_files"],
)

gentbl_cc_library(
    name = "xla_gpu_types_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_gpu_types.h.inc": [
            "-gen-typedef-decls",
            "-typedefs-dialect=xla_gpu",
        ],
        "xla_gpu_types.cc.inc": [
            "-gen-typedef-defs",
            "-typedefs-dialect=xla_gpu",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_gpu_types.td",
    deps = [":xla_gpu_td_files"],
)

cc_library(
    name = "xla_gpu",
    srcs = [
        "xla_gpu_attrs.cc",
        "xla_gpu_dialect.cc",
        "xla_gpu_ops.cc",
        "xla_gpu_types.cc",
    ],
    hdrs = ["xla_gpu_ops.h"],
    deps = [
        ":xla_gpu_attrs_inc_gen",
        ":xla_gpu_dialect_inc_gen",
        ":xla_gpu_ops_inc_gen",
        ":xla_gpu_types_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)
