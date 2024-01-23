load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("@local_tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "ops.mlir",
            "ops_verify.mlir",
            "testlib.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/runtime:xla-runtime-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

td_library(
    name = "testlib_td_files",
    srcs = [
        "testlib.td",
        "testlib_attrs.td",
        "testlib_enums.td",
        "testlib_types.td",
    ],
    compatible_with = get_compatible_with_portable(),
    includes = ["include"],
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "testlib_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "testlib_attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "testlib_attrs.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "testlib_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "testlib_dialect.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "testlib_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "testlib_enums.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "testlib_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "testlib_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "testlib.td",
    deps = [":testlib_td_files"],
)

cc_library(
    name = "testlib",
    srcs = ["testlib.cc"],
    hdrs = ["testlib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":testlib_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
