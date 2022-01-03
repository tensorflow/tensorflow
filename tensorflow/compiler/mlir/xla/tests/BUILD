load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
        "hlotxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
# This intentionally does not pull-in the top-level tf-opt to reduce the
# dependencies.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir/xla:xla-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

tf_cc_test(
    name = "mlir_hlo_builder_test",
    srcs = ["mlir_hlo_builder_test.cc"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/xla:mlir_hlo_builder",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
