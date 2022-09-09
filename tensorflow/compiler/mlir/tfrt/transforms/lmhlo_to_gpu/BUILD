load("//tensorflow:tensorflow.bzl", "if_google")

package(
    licenses = ["notice"],
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
#         "//tensorflow/compiler/mlir/xla:attribute_exporter",
#         "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
#         "//tensorflow/compiler/xla/mlir/transforms/gpu:passes",
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
