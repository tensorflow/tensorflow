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

cc_library(
    name = "FuncToGraph",
    srcs = ["func_to_graph.cc"],
    hdrs = ["func_to_graph.h"],
    deps = [
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        ":FuncToGraph",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms:PassIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//tensorflow/core/transforms:test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
