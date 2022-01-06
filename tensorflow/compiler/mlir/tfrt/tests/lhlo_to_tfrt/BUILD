load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "if_oss")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    default_tags = ["nomsan"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    tags_override = {
        "fusion.mlir": ["no_oss"],  # b/205277192
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/tfrt:lhlo-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
