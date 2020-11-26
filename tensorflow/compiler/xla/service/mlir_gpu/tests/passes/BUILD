load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    licenses = ["notice"],  # Apache 2.0
)

glob_lit_tests(
    data = [
        ":test_utilities",
        "@llvm-project//mlir:run_lit.sh",
    ],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/service/mlir_gpu:xla-mlir-gpu-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
