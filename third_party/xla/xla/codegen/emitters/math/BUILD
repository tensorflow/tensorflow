load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "ldexp",
    srcs = ["ldexp.cc"],
    hdrs = ["ldexp.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "ldexp_test",
    srcs = ["ldexp_test.cc"],
    deps = [
        ":ldexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",  # buildcleaner: keep
        "@llvm-project//llvm:ir_headers",
    ],
)
