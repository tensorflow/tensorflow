load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

td_library(
    name = "ifrt_td",
    srcs = [
        "ifrt_dialect.td",
        "ifrt_ops.td",
    ],
    visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ifrt_dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=ifrt",
            ],
            "ifrt_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=ifrt",
            ],
            "ifrt_dialect.cc.inc",
        ),
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=ifrt",
            ],
            "ifrt_types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=ifrt",
            ],
            "ifrt_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_dialect.td",
    test = True,
    deps = [":ifrt_td"],
)

gentbl_cc_library(
    name = "ifrt_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ifrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ifrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_ops.td",
    test = True,
    deps = [":ifrt_td"],
)

cc_library(
    name = "ir",
    srcs = [
        "ifrt_dialect.cc",
        "ifrt_ops.cc",
    ],
    hdrs = [
        "ifrt_dialect.h",
        "ifrt_ops.h",
    ],
    visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    deps = [
        ":ifrt_dialect_inc_gen",
        ":ifrt_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
