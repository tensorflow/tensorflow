# Description:
#   MLIR-GPU-specific components in XLA service implementation.

load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:friends"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "failover_compiler",
    srcs = ["failover_compiler.cc"],
    hdrs = ["failover_compiler.h"],
    deps = [
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "emission_context",
    srcs = ["emission_context.cc"],
    hdrs = ["emission_context.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "inject_errors_pass",
    srcs = ["inject_errors_pass.cc"],
    hdrs = ["inject_errors_pass.h"],
    deps = [
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "mlir_compiler",
    srcs = ["mlir_compiler.cc"],
    hdrs = ["mlir_compiler.h"],
    deps = [
        ":emission_context",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm-project//llvm:core",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
    ],
)

cc_library(
    name = "mlir_compiler_impl",
    srcs = if_cuda_is_configured(["mlir_compiler_impl.cc"]),
    deps = if_cuda_is_configured([
        ":mlir_compiler",
        ":failover_compiler",
        ":emission_context",
        ":kernel_lowering",
        ":lhlo_dialect_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetNVVMIR",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:gpu_constants",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_hlo_schedule",
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler_impl",
        "//tensorflow/compiler/xla/service/gpu:partition_assignment",
        "//tensorflow/compiler/xla/service/gpu:stream_assignment",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/gpu:asm_compiler",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "hlo_dialect_emitter",
    srcs = ["hlo_dialect_emitter.cc"],
    hdrs = ["hlo_dialect_emitter.h"],
    deps = [
        ":emission_context",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "lhlo_dialect_emitter",
    srcs = ["lhlo_dialect_emitter.cc"],
    hdrs = ["lhlo_dialect_emitter.h"],
    deps = [
        ":emission_context",
        ":hlo_dialect_emitter",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/mlir/xla:lhlo",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:thunk",
        "//tensorflow/compiler/xla/service/gpu:thunk_emitter",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "kernel_lowering",
    srcs = ["kernel_lowering.cc"],
    hdrs = ["kernel_lowering.h"],
    deps = [
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:hlo_legalize_to_lhlo",
        "//tensorflow/compiler/mlir/xla:lhlo",
        "//tensorflow/compiler/mlir/xla:lhlo_copy_removal",
        "//tensorflow/compiler/mlir/xla:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_gpu",
        "//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "//tensorflow/compiler/mlir/xla:xla_legalize_to_linalg",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "xla_gpu_opt_lib",
    testonly = True,
    srcs = ["xla_gpu_opt.cc"],
    hdrs = ["xla_gpu_opt.h"],
    tags = ["no_pip"],
    deps = [
        ":failover_compiler",
        ":inject_errors_pass",
        ":mlir_compiler",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_binary(
    name = "xla-gpu-opt",
    testonly = True,
    srcs = ["xla_gpu_opt_main.cc"],
    tags = ["no_pip"],
    deps = [
        ":mlir_compiler",
        ":xla_gpu_opt_lib",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:gpu_plugin_mlir",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
