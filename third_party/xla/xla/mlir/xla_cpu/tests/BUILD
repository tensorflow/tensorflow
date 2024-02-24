load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "bufferize.mlir",
            "invalid.mlir",
            "ops.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/backends/cpu:xla-cpu-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
