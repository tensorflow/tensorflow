load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-opt",
        "//tensorflow/compiler/mlir/tools/kernel_gen:kernel-gen-opt",
        "//tensorflow/compiler/xla/mlir_hlo:mlir-hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
