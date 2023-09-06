load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load("//xla:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# glob_lit_tests(
#     name = "all_tests",
#     data = [":test_utilities"],
#     default_tags = ["notap"],
#     driver = "//xla:run_lit.sh",
#     test_file_exts = ["mlir"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     data = [
#         "//xla/mlir/backends/gpu2:xla-gpu2-opt",
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//mlir:run_lit.sh",
#     ],
# )
#
# copybara:uncomment_end
