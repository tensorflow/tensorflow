load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# Custom `mlir-opt` replacement that links our dialect.
# The only purpose is testing the IR constructs themselves (parser, printer,
# verifiers...)
tf_native_cc_binary(
    name = "tfg-opt-no-passes",
    srcs = ["tfg-opt-no-passes.cc"],
    # Expose the test driver to core/ir.
    deps = [
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
    ],
)

# Bundle together all of the test utilities that are used by tests.
exports_files(["run_lit.sh"])

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-opt-no-passes",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
