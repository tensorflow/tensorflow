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
    tbl_outs = {
        "passes.h.inc": [
            "-gen-pass-decls",
            "-name=TritonXlaTransforms",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "extract_tma_info_pass.cc",
        "generalize_kernel_signature.cc",
        "int4_passes.cc",
        "round_f32_to_tf32_for_tf32_dot_pass.cc",
        "triton_xla_convert_unsupported_types.cc",
        "triton_xla_extract_insert_to_triton_pass.cc",
        "triton_xla_fold_transpose_pass.cc",
        "triton_xla_lower_atomics_pass.cc",
        "triton_xla_lower_block_barrier_pass.cc",
        "triton_xla_lower_get_tid_pass.cc",
        "triton_xla_squeeze_dims_pass.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla:permutation_util",
        "//xla:util",
        "//xla/backends/gpu/codegen/triton:emitter_helpers",
        "//xla/backends/gpu/codegen/triton/ir:triton_xla",
        "//xla/codegen:emitter_loc_op_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/gpu:tma_metadata",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@triton//:TritonDialects",
    ],
)
