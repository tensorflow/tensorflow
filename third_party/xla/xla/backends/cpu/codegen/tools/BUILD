load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "xla_cpu_opt",
    srcs = ["xla_cpu_opt.cc"],
    visibility = ["//xla/backends/cpu/codegen:__subpackages__"],
    deps = [
        "//xla/backends/cpu/codegen/ir:xla_cpu",
        "//xla/backends/cpu/codegen/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
