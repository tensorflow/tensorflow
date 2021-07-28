load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [
        ":test_utilities",
        "@llvm-project//mlir:run_lit.sh",
    ],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    tags_override = {
        # TODO(b/188517768): fix and enable these tests in OSS.
        "convert_ref_variables.mlir": ["no_oss"],
        "cross_device_transfer.mlir": ["no_oss"],
        "hoist_invariant_ops.mlir": ["no_oss"],
        "remote_run_encapsulate.mlir": ["no_oss"],
        "remove_device_attribute.mlir": ["no_oss"],
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir:tf-opt",
        "//tensorflow/compiler/mlir/tfrt:lhlo-tfrt-opt",
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
