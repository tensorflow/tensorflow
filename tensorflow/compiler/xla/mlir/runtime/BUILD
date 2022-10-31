load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package_group(
    name = "friends",
    packages = ["//tensorflow/compiler/xla/mlir/..."],
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "xla-runtime-opt",
    srcs = ["xla-runtime-opt.cc"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime/tests:testlib",
        "//tensorflow/compiler/xla/mlir/transforms/math:passes",
        "//tensorflow/compiler/xla/mlir/transforms/memref:passes",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:compilation_pipeline_cpu",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:compilation_pipeline_gpu",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
