load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["@triton//:license"],
    # default_compatible_with = ["//buildenv/target:non_prod"],
    # default_visibility = [
    # "@triton//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

td_library(
    name = "td_files",
    srcs = glob(["dialect/include/Dialect/Proton/IR/*.td"]),
    includes = ["dialect/include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "proton_ir_attr_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "dialect/include/Dialect/Proton/IR/ProtonAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "dialect/include/Dialect/Proton/IR/ProtonAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_ir_dialect_inc_gen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "dialect/include/Dialect/Proton/IR/Dialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "dialect/include/Dialect/Proton/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/include/Dialect/Proton/IR/ProtonDialect.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_ir_ops_inc_gen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "dialect/include/Dialect/Proton/IR/OpsEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "dialect/include/Dialect/Proton/IR/OpsEnums.cpp.inc",
        ),
        (
            ["--gen-op-decls"],
            "dialect/include/Dialect/Proton/IR/Ops.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "dialect/include/Dialect/Proton/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/include/Dialect/Proton/IR/ProtonOps.td",
    deps = ["td_files"],
)

cc_library(
    name = "ProtonIRDialect",
    srcs = glob([
        "dialect/lib/Dialect/Proton/IR/*.cpp",
    ]),
    hdrs = glob([
        "dialect/include/Dialect/Proton/IR/*.h",
    ]),
    includes = [
        "..",  # because proton/dialect/include/Dialect/Proton/IR/Dialect.h.inc
        "dialect/include",
    ],
    deps = [
        ":proton_ir_attr_inc_gen",
        ":proton_ir_dialect_inc_gen",
        ":proton_ir_ops_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TritonProtonToLLVM",
    srcs = glob([
        "dialect/lib/TritonProtonToLLVM/*.cpp",
    ]),
    hdrs = glob([
        "dialect/include/TritonProtonToLLVM/*.h",
    ]),
    includes = [
    ],
    deps = [
        ":ProtonIRDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
    ],
)
