load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

td_library(
    name = "rt_ops_td_files",
    srcs = [
        "rt_base.td",
        "rt_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "rt_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "rt_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "rt_dialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "rt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "rt_ops.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "rt_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "rt_types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rt_ops.td",
    deps = [":rt_ops_td_files"],
)

cc_library(
    name = "rt_ops",
    srcs = ["rt_ops.cc"],
    hdrs = ["rt_ops.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":rt_inc_gen",
        "//tensorflow/compiler/xla/runtime:constraints",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
