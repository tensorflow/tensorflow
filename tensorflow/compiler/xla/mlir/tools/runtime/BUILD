load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

tf_native_cc_binary(
    name = "xla-runtime-opt",
    srcs = ["xla_runtime_opt.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime/tests:testlib",
        "//tensorflow/compiler/xla/mlir/transforms/math:passes",
        "//tensorflow/compiler/xla/mlir/transforms/memref:passes",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:compilation_pipeline",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
