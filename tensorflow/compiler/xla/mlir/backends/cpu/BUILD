load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow/compiler/xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "xla-cpu-opt",
    srcs = ["xla-cpu-opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/mlir/backends/cpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_test_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:register",
    ],
)
