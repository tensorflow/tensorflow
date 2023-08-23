load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
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
    name = "dataflow_analysis",
    srcs = ["dataflow_analysis.cc"],
    hdrs = ["dataflow_analysis.h"],
    compatible_with = [],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "add_concurrent_regions.cc",
        "add_hlo_trace_annotations.cc",
        "gpu_to_gpu_runtime.cc",
        "lmhlo_gpu_to_gpu_runtime.cc",
        "lmhlo_to_gpu_launch.cc",
        "lmhlo_to_gpu_runtime.cc",
        "memref_get_global_to_arg.cc",
        "outline_cuda_graphs.cc",
        "passes.cc",
        "stream_assignment.cc",
        "uid_generator.h",
    ],
    hdrs = ["passes.h"],
    # Override cc_library()'s internal default value of ["//buildenv/target:gce"].`
    # TODO(ezhulenev): Do not depend on NCCL thunks in compiler passes.
    compatible_with = [],
    deps = [
        ":dataflow_analysis",
        ":passes_inc_gen",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/mlir/runtime/ir:rt",
        "//tensorflow/compiler/xla/mlir/runtime/utils:custom_calls",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/tsl/platform:env",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
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
