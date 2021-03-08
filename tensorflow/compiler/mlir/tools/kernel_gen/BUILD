load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "get_compatible_with_cloud",
    "tf_cc_binary",
)
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//tensorflow/core/platform:build_config.bzl", "if_llvm_aarch64_available", "if_llvm_system_z_available")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/core/kernels/mlir_generated/...",
    ],
)

cc_library(
    name = "kernel_creator",
    srcs = ["kernel_creator.cc"],
    hdrs = ["kernel_creator.h"],
    compatible_with = get_compatible_with_cloud(),
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_legalize_to_lhlo",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/hlo:legalize_trigonometric_to_approximation",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_gpu",
        "//tensorflow/compiler/mlir/hlo:transform_unranked_hlo",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "tf_to_kernel",
    srcs = ["tf_to_kernel.cc"],
    visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen/tests/tf_to_kernel:__pkg__",
        "//tensorflow/core/kernels/mlir_generated:__pkg__",
    ],
    deps = [
        ":kernel_creator",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ] + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]),
)

tf_cc_binary(
    name = "kernel-gen-opt",
    srcs = ["tools/kernel-gen-opt/kernel-gen-opt.cc"],
    visibility = ["//tensorflow/compiler/mlir/tools/kernel_gen/tests:__subpackages__"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

exports_files(["tf_framework_c_interface.h"])

cc_library(
    name = "tf_framework_c_interface",
    srcs = ["tf_framework_c_interface.cc"],
    hdrs = ["tf_framework_c_interface.h"],
    deps = [
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/core:framework",
        "@llvm-project//mlir:mlir_runner_utils",
    ],
)

cc_library(
    name = "tf_gpu_runtime_wrappers",
    deps = if_cuda_is_configured([
        ":tf_cuda_runtime_wrappers",
    ]) + if_rocm_is_configured([
        ":tf_rocm_runtime_wrappers",
    ]),
)

cc_library(
    name = "tf_cuda_runtime_wrappers",
    srcs = ["tf_cuda_runtime_wrappers.cc"],
    compatible_with = get_compatible_with_cloud(),
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",
        "//tensorflow/stream_executor:stream_header",
    ]),
)

cc_library(
    name = "tf_rocm_runtime_wrappers",
    srcs = if_rocm_is_configured(["tf_rocm_runtime_wrappers.cc"]),
    compatible_with = get_compatible_with_cloud(),
    copts = if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = if_rocm_is_configured([
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform/default/build_config:stream_executor_rocm",
        "//tensorflow/stream_executor:stream_header",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)
