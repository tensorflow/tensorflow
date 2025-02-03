load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "xla_cpu_td_files",
    srcs = glob(["*.td"]),
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_cpu_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "xla_cpu_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "xla_cpu_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_dialect.td",
    deps = [":xla_cpu_td_files"],
)

gentbl_cc_library(
    name = "xla_cpu_types_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=xla_cpu",
            ],
            "xla_cpu_types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=xla_cpu",
            ],
            "xla_cpu_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_types.td",
    deps = [":xla_cpu_td_files"],
)

gentbl_cc_library(
    name = "xla_cpu_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "xla_cpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xla_cpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_cpu_ops.td",
    deps = [":xla_cpu_td_files"],
)

cc_library(
    name = "xla_cpu",
    srcs = [
        "xla_cpu_dialect.cc",
        "xla_cpu_ops.cc",
        "xla_cpu_types.cc",
    ],
    hdrs = [
        "xla_cpu_dialect.h",
        "xla_cpu_ops.h",
        "xla_cpu_types.h",
    ],
    deps = [
        ":xla_cpu_dialect_inc_gen",
        ":xla_cpu_ops_inc_gen",
        ":xla_cpu_types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
