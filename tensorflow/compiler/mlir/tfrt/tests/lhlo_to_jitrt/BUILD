load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# copybara:uncomment_begin
#
# glob_lit_tests(
#     name = "all_tests",
#     data = [":test_utilities"],
#     driver = "//tensorflow/compiler/mlir:run_lit.sh",
#     features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
#     test_file_exts = ["mlir"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//llvm:not",
#         "@llvm-project//mlir:run_lit.sh",
#     ],
# )
#
# copybara:uncomment_end
