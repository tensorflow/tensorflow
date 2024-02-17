load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

build_test(
    name = "xla-cpu-opt_build_test",
    targets = [
        ":xla-cpu-opt",
    ],
)

xla_cc_binary(
    name = "xla-cpu-opt",
    srcs = ["xla-cpu-opt.cc"],
    deps = [
        "//xla/mlir/backends/cpu/transforms:passes",
        "//xla/mlir/xla_cpu/ir:xla_cpu",
        "//xla/mlir_hlo:all_passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/service/cpu:cpu_compiler",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
        "@stablehlo//:register",
    ],
)
