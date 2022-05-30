load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "if_oss")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    tags_override = {
        # tensorflow/compiler/mlir/tfrt:compatilibility_analysis is
        # intenionally not linked into tf-mlir-translate in OSS.
        "compatibility_analysis.mlir": ["no_oss"],
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
