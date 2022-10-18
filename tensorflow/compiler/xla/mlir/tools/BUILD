load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "xla-cpu-opt",
    srcs = ["xla_cpu_opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/mlir/transforms/cpu:passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_test_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:register",
    ],
)

tf_cc_binary(
    name = "xla-gpu-opt",
    srcs = ["xla_gpu_opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/mlir/transforms/gpu:passes",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

tf_cc_binary(
    name = "xla-runtime-opt",
    srcs = ["xla_runtime_opt.cc"],
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
