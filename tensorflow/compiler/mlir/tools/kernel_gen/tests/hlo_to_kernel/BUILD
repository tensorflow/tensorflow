load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = [
        # We need access to the CUDA SDK.
        "gpu",
        "no_rocm",
    ],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:hlo_to_kernel",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
