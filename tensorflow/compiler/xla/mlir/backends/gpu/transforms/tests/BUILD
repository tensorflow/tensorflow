load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/compiler/xla:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # TODO(b/286919981). Remove nomsan once we pass through MSAN_OPTIONS env var
    # to the test.
    default_tags = ["nomsan"],
    driver = "//tensorflow/compiler/xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/mlir/backends/gpu:xla-gpu-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
