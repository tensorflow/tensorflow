load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [
        "load-quantization-recipe.mlir",
        # TODO(b/493282789): Enable DenseResourceElementsAttr constant folding with
        # more peformant folder implementations.
        "const-fold-unary-resource.mlir",
    ],
    size_override = {
        "gpu_compatibility.mlir": "small",
        "optimize.mlir": "large",
        "prepare-tf.mlir": "medium",
        "prepare-tf-fake-quant.mlir": "medium",
        "prepare-tf-fake-quant-4bit.mlir": "medium",
        "raise-custom-ops.mlir": "medium",
        "rank_reduction.mlir": "small",
    },
    tags_override = {
        "legalize-tf.mlir": ["no_rocm"],
        "optimize.mlir": ["no_rocm"],
        "prepare-tf.mlir": ["no_rocm"],
        "const-fold.mlir": ["no_mac_arm64"],
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/lite:litert-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
