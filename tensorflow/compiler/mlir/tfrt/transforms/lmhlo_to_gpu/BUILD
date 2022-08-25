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
        # copybara:uncomment ":lmhlo_to_jitrt",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

# copybara:uncomment_begin
#
# cc_library(
#     name = "lmhlo_to_jitrt",
#     srcs = ["lmhlo_to_jitrt.cc"],
#     hdrs = ["lmhlo_to_jitrt.h"],
#     tags = ["gpu"],
#     visibility = [
#         "//tensorflow/compiler/mlir/tfrt:__pkg__",
#         "//tensorflow/compiler/xla/service/gpu:__pkg__",
#     ],
#     deps = [
#         ":JitRtPassesIncGen",
#         ":lmhlo_to_gpu_binary",
#         "//tensorflow/compiler/mlir/xla:attribute_exporter",
#         "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
#         "//tensorflow/compiler/xla/mlir_hlo:sink_constants_to_control_flow",
#         "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
#         "@llvm-project//mlir:ArithmeticDialect",
#         "@llvm-project//mlir:ControlFlowDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:GPUDialect",
#         "@llvm-project//mlir:GPUTransforms",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:SCFDialect",
#         "@llvm-project//mlir:TransformUtils",
#         "@llvm-project//mlir:Transforms",
#         "@tf_runtime//backends/gpu:gpu_opdefs",
#         "@tf_runtime//backends/gpu:gpu_passes",
#     ],
# )
#
# copybara:uncomment_end

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
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
        "//tensorflow/compiler/xla/service/gpu:gpu_device_info",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:nvptx_helper",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu:thunk",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/stream_executor:device_description",
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
