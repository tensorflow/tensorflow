load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "hlo_text_to_lhlo_no_opt.hlotxt",
            "non_identity_layouts.hlotxt",
            "no_opt_ops.hlotxt",
        ],
        include = [
            "*.hlotxt",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/translate:xla-translate",
        "//xla/translate/mhlo_to_lhlo_with_xla:xla-translate-gpu-opt",
        "//xla/translate/mhlo_to_lhlo_with_xla:xla-translate-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
