load("//xla:xla.default.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

xla_cc_test(
    name = "tensor_or_memref_test",
    srcs = ["tensor_or_memref_test.cc"],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "interpreter_value_test",
    srcs = ["interpreter_value_test.cc"],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
