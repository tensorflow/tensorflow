load("//tensorflow/compiler/xla:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/mlir_hlo:mlir-interpreter-runner",
        "@llvm-project//llvm:FileCheck",
    ],
)
