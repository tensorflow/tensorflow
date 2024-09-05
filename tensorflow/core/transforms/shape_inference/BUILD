load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [
        "//tensorflow/core/transforms:__subpackages__",
    ],
    licenses = ["notice"],  # apache 2.0
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir:shape_inference_utils",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/transforms:PassIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//tensorflow/core/transforms:test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
