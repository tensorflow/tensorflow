load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

lit_test_suite(
    name = "mlir_lit_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/service/gpu/tests:xla-opt",
        "@llvm-project//llvm:FileCheck",
        "@triton//:triton-opt",
    ],
)
