load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load("//xla:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
        "hlo",
        "hlotxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//xla/translate:xla-translate",
        "//xla/translate/mhlo_to_lhlo_with_xla:xla-translate-gpu-opt",
        "//xla/translate/mhlo_to_lhlo_with_xla:xla-translate-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
    visibility = ["//visibility:public"],
)
