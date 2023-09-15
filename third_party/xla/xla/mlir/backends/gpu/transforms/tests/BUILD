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
    # TODO(b/286919981). Remove nomsan once we pass through MSAN_OPTIONS env var
    # to the test.
    default_tags = ["nomsan"],
    driver = "//xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//xla/mlir/backends/gpu:xla-gpu-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
    visibility = ["//visibility:public"],
)
