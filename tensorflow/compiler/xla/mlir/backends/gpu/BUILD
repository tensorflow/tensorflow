load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "xla-gpu-opt",
    srcs = ["xla-gpu-opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/mlir/backends/gpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
