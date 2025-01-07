load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tests:build_defs.bzl", "xla_test")

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
    name = "xla_td_files",
    srcs = glob(["*.td"]),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_dialect_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "xla_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "xla_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_dialect.td",
    deps = [":xla_td_files"],
)

gentbl_cc_library(
    name = "xla_ops_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "xla_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xla_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_ops.td",
    deps = [":xla_td_files"],
)

gentbl_cc_library(
    name = "xla_attrs_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "xla_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "xla_enums.cc.inc",
        ),
        (
            [
                "-gen-attrdef-decls",
            ],
            "xla_attrs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "xla_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_attrs.td",
    deps = [":xla_td_files"],
)

cc_library(
    name = "xla",
    srcs = [
        "xla_attrs.cc",
        "xla_dialect.cc",
        "xla_ops.cc",
    ],
    hdrs = ["xla_ops.h"],
    deps = [
        ":xla_attrs_inc_gen",
        ":xla_dialect_inc_gen",
        ":xla_ops_inc_gen",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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

xla_test(
    name = "xla_ops_test",
    srcs = ["xla_ops_test.cc"],
    backends = ["cpu"],
    deps = [
        ":xla",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:test",
    ],
)
