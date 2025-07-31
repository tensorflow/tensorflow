load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # apache 2.0
)

gentbl_cc_library(
    name = "MklPDLLPatternsIncGen",
    tbl_outs = {"pdll/MklPDLLPatterns.h.inc": ["-x=cpp"]},
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
    hdrs = [
        "pass.h",
        "remapping_helper.h",
    ],
    deps = [
        ":MklPDLLPatternsIncGen",
        "//tensorflow/core:framework",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms:PassIncGen",
        "//tensorflow/core/transforms:op_cat_helper",
        "//tensorflow/core/transforms:pdll_utils",
        "//tensorflow/core/transforms:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//tensorflow/core/transforms:test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)
