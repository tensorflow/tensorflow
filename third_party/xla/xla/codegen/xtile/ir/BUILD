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
    name = "xtile_td_files",
    srcs = glob(["*.td"]),
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BufferizableOpInterfaceTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xtile_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xtile_dialect.h.inc": ["-gen-dialect-decls"],
        "xtile_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xtile_dialect.td",
    deps = [":xtile_td_files"],
)

gentbl_cc_library(
    name = "xtile_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xtile_interface_ops.h.inc": ["-gen-op-interface-decls"],
        "xtile_interface_ops.cc.inc": ["-gen-op-interface-defs"],
        "xtile_ops.h.inc": ["-gen-op-decls"],
        "xtile_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xtile_ops.td",
    deps = [":xtile_td_files"],
)

gentbl_cc_library(
    name = "xtile_attrs_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xtile_attrs.h.inc": ["-gen-attrdef-decls"],
        "xtile_attrs.cc.inc": ["-gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xtile_attrs.td",
    deps = [":xtile_td_files"],
)

cc_library(
    name = "xtile",
    srcs = [
        "xtile_attrs.cc",
        "xtile_bufferization.cc",
        "xtile_dialect.cc",
        "xtile_ops.cc",
    ],
    hdrs = [
        "xtile_attrs.h",
        "xtile_dialect.h",
        "xtile_ops.h",
    ],
    deps = [
        ":xtile_attrs_inc_gen",
        ":xtile_dialect_inc_gen",
        ":xtile_ops_inc_gen",
        "//xla/codegen/emitters:implicit_arith_op_builder",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)
