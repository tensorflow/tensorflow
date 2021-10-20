load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

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
        "//tensorflow/compiler/mlir/tfjs:tfjs-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
