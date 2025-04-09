load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=GpuFusionTransforms",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "convert_float_amd.cc",
        "convert_float_nvidia.cc",
        "convert_index_type.cc",
        "fuse_loops.cc",
        "optimize_loops.cc",
        "peel_loops.cc",
    ],
    hdrs = ["passes.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":passes_inc_gen",
        "//xla:util",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/transforms:atomic_rmw_utils",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_cuda_is_configured([
        "//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu/llvm_gpu_backend:amdgpu_backend",
    ]),
)
