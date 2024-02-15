load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "compilation_pipeline.mlir",
            "convert_asserts.mlir",
            "convert_custom_calls.mlir",
            "export_functions.mlir",
            "move_allocas_to_entry_block.mlir",
            "ordinal_assignment.mlir",
            "rt_to_llvm.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/runtime:xla-runtime-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

cc_library(
    name = "testlib_pipeline",
    testonly = 1,
    srcs = ["testlib_pipeline.cc"],
    hdrs = ["testlib_pipeline.h"],
    visibility = ["//xla:runtime"],
    deps = [
        "//xla/mlir/runtime/transforms:compiler",
        "//xla/mlir/runtime/transforms:passes",
        "//xla/runtime:compiler",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
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
