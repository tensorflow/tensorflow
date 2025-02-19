load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "dummy_passes",
    hdrs = ["dummy_passes.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            "algebraic_simplifier.hlo",
            "validate_registered_passes_runs.hlo",
            "hlo_opt_expect_failure.hlo",
            "run_multiple_passes.hlo",
            "run_single_pass.hlo",
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
