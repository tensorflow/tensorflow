load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GpuTransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "gpu_to_gpu_runtime.cc",
        "lmhlo_gpu_to_gpu_runtime.cc",
        "lmhlo_to_gpu_launch.cc",
        "lmhlo_to_gpu_runtime.cc",
        "memref_get_global_to_arg.cc",
        "passes.cc",
    ],
    hdrs = ["passes.h"],
    # Override cc_library()'s internal default value of ["//buildenv/target:gce"].`
    # TODO(ezhulenev): Do not depend on NCCL thunks in compiler passes.
    compatible_with = [],
    deps = [
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/utils/runtime:custom_calls",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
