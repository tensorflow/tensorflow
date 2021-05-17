load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//third_party/mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:friends",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "tf_framework_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
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
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_framework_ops.td",
    td_srcs = [
        "tf_status.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_status_inc_gen",
    compatible_with = get_compatible_with_cloud(),
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
    td_srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
    ],
)
