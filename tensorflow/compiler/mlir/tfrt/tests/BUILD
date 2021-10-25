load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "if_oss")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    tags_override = {
        "lhlo_to_tfrt/async_conversion.mlir": ["nomsan"],
        "lhlo_to_tfrt/blas.mlir": ["nomsan"],
        "lhlo_to_tfrt/ccl.mlir": ["nomsan"],
        "lhlo_to_tfrt/cholesky.mlir": ["nomsan"],
        "lhlo_to_tfrt/common.mlir": ["nomsan"],
        "lhlo_to_tfrt/custom_call.mlir": ["nomsan"],
        "lhlo_to_tfrt/fusion.mlir": ["nomsan"],
        "lhlo_to_tfrt/memcpy.mlir": ["nomsan"],
        "lhlo_to_tfrt/memset.mlir": ["nomsan"],
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
        "@llvm-project//mlir:run_lit.sh",
    ],
)
