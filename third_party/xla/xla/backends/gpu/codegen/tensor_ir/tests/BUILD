load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "lit_tests",
    srcs = glob(["**/*.hlo"]),
    cfg = "//xla:lit.cfg.py",
    default_tags = [
        "cuda-only",
        "gpu",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [
        "//xla/backends/gpu/codegen/tools:fusion_to_tensorir",
        "@llvm-project//llvm:FileCheck",
    ],
)
