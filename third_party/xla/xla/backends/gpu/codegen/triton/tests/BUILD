load("//xla:lit.bzl", "lit_test_suite")  # @unused

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

# copybara:uncomment_begin(triton-opt tool doesn't build in OSS)
# lit_test_suite(
#     name = "mlir_lit_tests",
#     srcs = glob(["*.mlir"]),
#     cfg = "//xla:lit.cfg.py",
#     tools = [
#         "@llvm-project//llvm:FileCheck",
#         "//xla/service/gpu/tests:xla-opt",
#     ],
# )
# copybara:uncomment_end
