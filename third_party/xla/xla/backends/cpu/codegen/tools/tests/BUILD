load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "ir_compiler_opt_tests",
    srcs = [
        "ir_compiler_basic_test.ll",
        "ir_compiler_vectorization_test.ll",
    ],
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/backends/cpu/codegen/tools:ir-compiler-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
