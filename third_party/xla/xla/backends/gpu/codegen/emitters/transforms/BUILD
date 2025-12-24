load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
        "lower_xla_shared.cc",
        "optimize_loops.cc",
        "peel_loops.cc",
        "recover_exp2.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/service/gpu:gpu_fusible",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
    ],
)
