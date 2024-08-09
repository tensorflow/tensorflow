load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "tests",
    srcs = glob(["**/*.hlo"]),
    cfg = "//xla:lit.cfg.py",
    default_tags = ["requires-gpu-sm80-only"],
    tools = [
        "//xla/service/gpu/fusions/tools:fusion_to_mlir",
        "//xla/service/gpu/fusions/tools:mlir_fusions_opt",
        "//xla/service/gpu/fusions/tools:test_correctness",
        "@llvm-project//llvm:FileCheck",
    ],
)
