load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/hlo:mlir-hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

cc_library(
    name = "capi_test",
    srcs = ["capi_test.c"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:CAPI",
    ],
)

build_test(
    name = "capi_build_test",
    testonly = True,
    targets = ["capi_test"],
)
