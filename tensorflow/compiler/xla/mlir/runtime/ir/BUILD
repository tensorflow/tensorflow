load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/mlir/runtime:friends"],
    licenses = ["notice"],
)

td_library(
    name = "rt_ops_td_files",
    srcs = [
        "rt_dialect.td",
        "rt_interfaces.td",
        "rt_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
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
            "rt_dialect.cc.inc",
        ),
        (
            ["-gen-op-decls"],
            "rt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "rt_ops.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "rt_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "rt_types.cc.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "rt_attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "rt_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rt_ops.td",
    deps = [":rt_ops_td_files"],
)

gentbl_cc_library(
    name = "rt_interfaces_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "rt_attr_interfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "rt_attr_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rt_interfaces.td",
    deps = [":rt_ops_td_files"],
)

cc_library(
    name = "rt",
    srcs = [
        "rt_dialect.cc",
        "rt_interfaces.cc",
        "rt_ops.cc",
    ],
    hdrs = [
        "rt_dialect.h",
        "rt_interfaces.h",
        "rt_ops.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":rt_inc_gen",
        ":rt_interfaces_inc_gen",
        "//tensorflow/compiler/xla/runtime:constraints",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)
