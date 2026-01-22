load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "hlo_extractor_tests",
    srcs = ["hlo_extractor_test.hlo"],
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/tools:hlo-extractor",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "hlo_expand_test",
    srcs = ["hlo_expand_test.cc"],
    data = [
        "data/cholesky.hlo",
        "data/invalid_concat.hlo",
        "data/spmd.hlo",
        "//xla/tools:hlo-expand",
    ],
    tags = [
        "nomsan",  # No msan for precompiled Nvidia binaries.
    ],
    deps = [
        "//xla/tsl/platform:subprocess",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
    ],
)
