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
    tags_override = {
        "triton_xla_math_to_libdevice_rocm.mlir": ["rocm-only"],
    },
    tools = [
        "//xla/backends/gpu/tests:xla-opt",
        "@llvm-project//llvm:FileCheck",
        "@triton//:triton-opt",
    ],
)
