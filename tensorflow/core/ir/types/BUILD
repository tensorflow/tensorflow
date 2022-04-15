load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = ["//tensorflow/core:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "attributes.td",
        "dialect.td",
        "types.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":DialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "attributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "attributes.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "attributes_enum.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "attributes_enum.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "TypesIncGen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "types.td",
    deps = [":DialectTdFiles"],
)

cc_library(
    name = "Dialect",
    srcs = ["dialect.cc"],
    hdrs = [
        "dialect.h",
    ],
    textual_hdrs = [
        "types.def",
    ],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":AttributesIncGen",
        ":DialectIncGen",
        ":TypesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "dialect_test",
    size = "small",
    srcs = ["dialect_test.cc"],
    deps = [
        ":Dialect",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)
