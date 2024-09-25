# Lit tests for the Shardy in XLA.

load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "shardy_lit_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/service/spmd/shardy:sdy_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
