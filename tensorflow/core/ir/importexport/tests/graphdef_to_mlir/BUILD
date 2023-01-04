load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# Bundle together all of the test utilities that are used by tests.
exports_files(["run_lit.sh"])

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/core/ir/importexport:tfg-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = [
        "mlir",
        "pbtxt",
    ],
)
