load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow:tensorflow.bzl", "if_google")

# TF to TFRT kernels conversion.
package(
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "GpuPassesIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [(
        ["-gen-pass-decls"],
        "gpu_passes.h.inc",
    )],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gpu_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "JitRtPassesIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [(
        ["-gen-pass-decls"],
        "jitrt_passes.h.inc",
    )],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jitrt_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "pattern_utils",
    srcs = ["pattern_utils.cc"],
    hdrs = ["pattern_utils.h"],
    tags = ["gpu"],
    deps = [
        "//tensorflow/stream_executor:dnn",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
)

cc_library(
    name = "pass_utils",
    srcs = ["pass_utils.cc"],
    hdrs = ["pass_utils.h"],
    tags = ["gpu"],
    visibility = if_google([
        "//platforms/xla/tests/gpu:__pkg__",
    ]) + [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        ":lmhlo_to_gpu",
        ":lmhlo_to_gpu_binary",
        ":lmhlo_to_jitrt",
        ":lmhlo_to_tfrt_gpu",
        ":pattern_utils",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tf_runtime//backends/gpu:gpu_passes",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
)

cc_library(
    name = "lmhlo_to_gpu",
    srcs = [
        "ccl_pattern.cc",
        "cholesky_pattern.cc",
        "convolution_pattern.cc",
        "custom_call_pattern.cc",
        "fft_pattern.cc",
        "gemm_pattern.cc",
        "infeed_and_outfeed_pattern.cc",
        "lmhlo_to_gpu.cc",
        "lmhlo_to_gpu_branch.cc",
        "replica_and_partition_pattern.cc",
        "triangular_solve_pattern.cc",
    ],
    hdrs = [
        "lmhlo_to_gpu.h",
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
        ":GpuPassesIncGen",
        ":pattern_utils",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/xla:attribute_exporter",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/gpu:gpu_conv_runner",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:matmul_utils",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "//tensorflow/compiler/xla/service/gpu:xlir_opdefs",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_passes",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lmhlo_to_tfrt_gpu",
    srcs = ["lmhlo_to_tfrt_gpu.cc"],
    hdrs = ["lmhlo_to_tfrt_gpu.h"],
    tags = ["gpu"],
    visibility = if_google([
        "//platforms/xla/tests/gpu:__pkg__",
    ]) + [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        ":lmhlo_to_gpu",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_passes",
    ],
)

cc_library(
    name = "lmhlo_to_jitrt",
    srcs = ["lmhlo_to_jitrt.cc"],
    hdrs = ["lmhlo_to_jitrt.h"],
    tags = ["gpu"],
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        ":JitRtPassesIncGen",
        ":lmhlo_to_gpu_binary",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_passes",
        "@tf_runtime//backends/jitrt:rt_opdefs",
    ],
)

# The lmhlo-to-gpu-binary pass is a separate target to avoid a dependency cycle:
# :lmhlo_to_gpu_binary > xla/service/gpu:gpu_executable > :lmhlo_to_gpu
cc_library(
    name = "lmhlo_to_gpu_binary",
    srcs = [
        "kernel_ops_pattern.cc",
        "lmhlo_to_gpu_binary.cc",
    ],
    hdrs = [
        "lmhlo_to_gpu_binary.h",
    ],
    tags = ["gpu"],
    visibility = if_google([
        "//platforms/xla/tests/gpu:__pkg__",
    ]) + [
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/compiler/xla/service/gpu:__pkg__",
    ],
    deps = [
        ":GpuPassesIncGen",
        ":pattern_utils",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:nvptx_helper",
        "//tensorflow/compiler/xla/service/gpu:thunk",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//backends/gpu:gpu_passes",
    ],
    alwayslink = 1,
)
