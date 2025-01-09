load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    data = ["ifrt_legalize_to_vifrt.0_1_0.mlir.bytes"],
    tools = [
        "//xla/python/ifrt/ir/tests:ifrt-opt",
        "//xla/python/ifrt/ir/tests:ifrt-translate",
        "@llvm-project//llvm:FileCheck",
    ],
)
