load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    tags_override = {
        "gpu_ops.mlir": tf_cuda_tests_tags() + [
            "noasan",
            "nomsan",
            "noubsan",
            "no_cuda_asan",
            "no_oss",
        ],  # b/171751580
        "ops.mlir": tf_cuda_tests_tags() + [
            "nomsan",
            "no_cuda_asan",
            "no_oss",
        ],  # b/191025174, b/192521861
        "passthrough.mlir": tf_cuda_tests_tags() + [
            "nomsan",
            "no_cuda_asan",
            "no_cuda_asan",
            "no_oss",
        ],  # b/191025174, b/192521861
    },
    test_file_exts = [
        "mlir",
        "hlo",
        "hlotxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir/xla:xla-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
