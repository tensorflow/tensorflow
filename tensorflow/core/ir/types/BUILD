load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_portable(),
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
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = {
        "dialect.h.inc": ["-gen-dialect-decls"],
        "dialect.cpp.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":DialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = {
        "attributes.h.inc": ["-gen-attrdef-decls"],
        "attributes.cc.inc": ["-gen-attrdef-defs"],
        "attributes_enum.h.inc": ["-gen-enum-decls"],
        "attributes_enum.cc.inc": ["-gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "TypesIncGen",
    tbl_outs = {
        "types.h.inc": ["-gen-typedef-decls"],
        "types.cc.inc": ["-gen-typedef-defs"],
    },
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
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":AttributesIncGen",
        ":DialectIncGen",
        ":TypesIncGen",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
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
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
