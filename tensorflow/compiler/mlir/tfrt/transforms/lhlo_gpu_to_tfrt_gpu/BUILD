load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow:tensorflow.bzl", "if_google")

# TF to TFRT kernels conversion.
package(
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "TfrtGpuPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Xlir",
            ],
            "gpu_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gpu_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "ccl_pattern.cc",
        "custom_call_pattern.cc",
        "gemm_pattern.cc",
        "gpu_passes.cc",
    ],
    hdrs = [
        "gpu_passes.h",
        "register_passes.h",
    ],
    tags = [
        "gpu",
        "no_oss",
    ],
    visibility = if_google([
        "//platforms/xla/tests/gpu:__pkg__",
    ]) + [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        ":TfrtGpuPassIncGen",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "//tensorflow/compiler/xla/service/gpu:xlir_opdefs",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_passes",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
    alwayslink = 1,
)
