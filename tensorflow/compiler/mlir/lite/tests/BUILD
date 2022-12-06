load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = ["load-quantization-recipe.mlir"],
    size_override = {
        "optimize.mlir": "medium",
        "prepare-tf.mlir": "medium",
        "prepare-tf-fake-quant.mlir": "medium",
        "prepare-tf-fake-quant-4bit.mlir": "medium",
        "raise-custom-ops.mlir": "medium",
    },
    tags_override = {
        "legalize-tf.mlir": ["no_rocm"],
        "optimize.mlir": ["no_rocm"],
        "prepare-tf.mlir": ["no_rocm"],
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
