load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir/lite/stablehlo:run_lit.sh",
    test_file_exts = [
        "mlir",
        "cc",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate",
        "//tensorflow/compiler/mlir/lite/stablehlo:odml_to_stablehlo",
        "//tensorflow/compiler/mlir/lite/stablehlo:tf-mhlo-tfl-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
