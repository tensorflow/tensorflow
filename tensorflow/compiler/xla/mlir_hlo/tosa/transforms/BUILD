# Legalizations and transforms for MHLO -> TOSA.

load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":internal",
        "//tensorflow/compiler/xla/mlir_hlo/tosa:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [],
)

gentbl_cc_library(
    name = "MHLOTOSAPDLLPatternsIncGen",
    tbl_outs = [
        (
            ["-x=cpp"],
            "legalize_mhlo/legalize_mhlo.pdll.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "legalize_mhlo/legalize_mhlo.pdll",
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TosaDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "MHLOTOSATransformsPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MHLOTOSATransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MHLOTOSATransforms",
    srcs = [
        "legalize_mhlo/legalize_mhlo.cc",
        "prepare_mhlo/prepare_mhlo.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    includes = ["."],
    deps = [
        ":MHLOTOSAPDLLPatternsIncGen",
        ":MHLOTOSATransformsPassIncGen",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
