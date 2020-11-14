# Description:
#   Integration tests of converter & interpreter.
#
# There should be few tests in here and it should be rare where the execution
# tests are not tested by unit tests already. This is useful for verifying some
# runtime behavior, but the majority of runtime tests should be TFLite side and
# invariants only verified in the converter/compiler.

load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

licenses(["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-opt",
        "//tensorflow/compiler/mlir/lite:mlir-tflite-runner",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
