load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "lit_tests",
    srcs = glob(
        include = ["**/*.hlo"],
        exclude = [],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/backends/cpu/codegen/tiled/tools:fusion_to_xtile",
        "//xla/backends/cpu/codegen/tiled/tools:test_correctness",
        "//xla/backends/cpu/codegen/tools:fusion_compiler_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
