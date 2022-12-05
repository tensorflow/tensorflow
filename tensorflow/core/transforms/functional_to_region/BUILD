load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # apache 2.0
)

cc_library(
    name = "Impl",
    srcs = ["impl.cc"],
    hdrs = ["impl.h"],
    deps = [
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/transforms:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        ":Impl",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms:PassIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

glob_lit_tests(
    data = ["//tensorflow/core/transforms:test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
