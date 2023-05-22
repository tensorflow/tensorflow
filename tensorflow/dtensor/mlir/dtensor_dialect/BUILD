# DTensor MLIR dialect.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "ir/dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "ir/dialect.cc.inc",
        ),
    ],
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
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/dtensor/cc:tensor_layout",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
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
    ],
)
