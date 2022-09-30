load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow:internal"],
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
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_tiling_cwise",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_tiling_reduction",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:thlo",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
    ],
)
