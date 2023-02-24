load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir/lite/stablehlo:run_lit.sh",
    size_override = {
        "legalize-skip-quantization-ops.mlir": "medium",
    },
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
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
        "//tensorflow/compiler/mlir/lite/stablehlo:odml-to-stablehlo-opt",
        "//tensorflow/compiler/mlir/lite/stablehlo:odml_to_stablehlo",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
