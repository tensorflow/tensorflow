load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            # go/keep-sorted start
            "algebraic_simplifier.hlo",
            "cholesky_expander.hlo",
            "operand_upcaster.hlo",
            "optimization_barrier_expander.hlo",
            "rewrite_bf16_conv_to_onednn.hlo",
            "rng_bit_generator_expander.hlo",
            "rng_expander.hlo",
            # go/keep-sorted end
        ],
        include = [
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/hlo/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
