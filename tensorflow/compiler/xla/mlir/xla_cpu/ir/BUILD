load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "td_files",
    srcs = [
        "xla_cpu_dialect.td",
        "xla_cpu_enums.td",
        "xla_cpu_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//mlir:BufferizableOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_cpu_dialect_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "xla_cpu_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "xla_cpu_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_dialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "xla_cpu_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "xla_cpu.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xla_cpu.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_ops.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "xla_cpu_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "xla_cpu_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "xla_cpu_enums.cc.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "xla_cpu_attrdefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "xla_cpu_attrdefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_enums.td",
    deps = [
        ":td_files",
    ],
)

cc_library(
    name = "xla_cpu",
    srcs = [
        "xla_cpu.cc",
    ],
    hdrs = ["xla_cpu.h"],
    deps = [
        ":xla_cpu_dialect_inc_gen",
        ":xla_cpu_enums_inc_gen",
        ":xla_cpu_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
