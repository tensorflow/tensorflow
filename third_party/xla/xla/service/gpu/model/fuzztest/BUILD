load(
    "@com_google_fuzztest//build_defs:cc_fuzztest_grammar_library.bzl",
    "cc_fuzztest_grammar_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_fuzztest_grammar_library(
    name = "affine_grammar",
    srcs = ["affine_fuzz.g4"],
    top_level_rule = "affine",
)

cc_test(
    name = "affine_simplifier_fuzz_test",
    srcs = ["affine_simplifier_fuzz_test.cc"],
    deps = [
        ":affine_grammar",
        "//xla/service/gpu/model:affine_map_printer",
        "//xla/service/gpu/model:indexing_analysis",
        "//xla/service/gpu/model:indexing_test_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
