load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:friends",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "td_files",
    srcs = [
        "tf_framework_ops.td",
        "tf_status.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:AllocationOpInterfaceTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_framework_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_framework_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_framework_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "tf_framework_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "tf_framework_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_framework_ops.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "tf_status_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "tf_status.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "tf_status.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_status.td",
    deps = [":td_files"],
)

cc_library(
    name = "tf_framework_ops",
    srcs = [
        "tf_framework_ops.cc",
        "tf_framework_ops.cc.inc",
        "tf_framework_ops.h.inc",
    ],
    hdrs = ["tf_framework_ops.h"],
    deps = [
        ":tf_framework_ops_inc_gen",
        ":tf_status_inc_gen",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:AllocationOpInterface",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
