load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # apache 2.0
)

gentbl_cc_library(
    name = "MklPDLLPatternsIncGen",
    tbl_outs = [
        (
            ["-x=cpp"],
            "pdll/MklPDLLPatterns.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "pdll/mkl_patterns.pdll",
    deps = [
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms:PDLLUtilsFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        ":MklPDLLPatternsIncGen",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms:PassDetail",
        "//tensorflow/core/transforms:pdll_utils",
        "//tensorflow/core/transforms:utils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

glob_lit_tests(
    data = ["//tensorflow/core/transforms:test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)
