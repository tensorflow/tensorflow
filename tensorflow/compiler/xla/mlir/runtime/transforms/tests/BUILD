load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/mlir/runtime:xla-runtime-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

cc_library(
    name = "testlib_pipeline",
    testonly = 1,
    srcs = ["testlib_pipeline.cc"],
    hdrs = ["testlib_pipeline.h"],
    visibility = ["//tensorflow/compiler/xla:runtime"],
    deps = [
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:passes",
        "//tensorflow/compiler/xla/runtime:compiler",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",
    ],
)
