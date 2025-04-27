load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

# DTensor MLIR dialect.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ],
    licenses = ["notice"],
)

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.

td_library(
    name = "dtensor_td_files",
    srcs = [
        "ir/dtensor_dialect.td",
        "ir/dtensor_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/ops.h.inc": ["-gen-op-decls"],
        "ir/ops.cc.inc": ["-gen-op-defs"],
        "ir/dialect.h.inc": ["-gen-dialect-decls"],
        "ir/dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/dtensor_ops.td",
    deps = [":dtensor_td_files"],
)

cc_library(
    name = "Dialect",
    srcs = ["ir/ops.cc"],
    hdrs = [
        "ir/dialect.h",
        "ir/ops.h",
    ],
    deps = [
        ":DialectIncGen",
        ":ir/dtensor_attributes",
        "//tensorflow/dtensor/cc:dstatus",
        "//tensorflow/dtensor/cc:tensor_layout",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ir/dtensor_attributes",
    srcs = ["ir/dtensor_attributes.cc"],
    hdrs = [
        "ir/dialect.h",
        "ir/dtensor_attributes.h",
    ],
    deps = [
        ":DialectIncGen",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
