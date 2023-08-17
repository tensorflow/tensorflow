load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/compiler/xla:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# glob_lit_tests(
#     data = [":test_utilities"],
#     default_tags = ["notap"],
#     driver = "//tensorflow/compiler/xla:run_lit.sh",
#     test_file_exts = ["mlir"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//mlir:run_lit.sh",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2:xla-gpu2-opt",
#     ],
# )
#
# copybara:uncomment_end
