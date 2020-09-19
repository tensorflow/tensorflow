load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

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
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_legalize_to_lhlo",
        "//tensorflow/compiler/mlir/hlo:legalize_tanh_to_approximation",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_gpu",
        "//tensorflow/compiler/mlir/hlo:materialize_broadcasts",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/hlo:transform_unranked_hlo",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/hlo:unfuse_batch_norm",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/mlir_gpu:kernel_lowering",
        "//tensorflow/compiler/xla/service/mlir_gpu:passes",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetNVVMIR",
        "@llvm-project//mlir:TargetROCDLIR",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "tf_to_gpu_binary",
    srcs = ["tf_to_gpu_binary.cc"],
    visibility = ["//tensorflow/core/kernels/mlir_generated:__pkg__"],
    deps = [
        ":kernel_creator",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_binary(
    name = "tf_to_kernel",
    srcs = ["tf_to_kernel.cc"],
    visibility = ["//tensorflow/core/kernels/mlir_generated:__pkg__"],
    deps = [
        ":kernel_creator",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)

tf_cc_binary(
    name = "kernel-gen-opt",
    srcs = ["tools/kernel-gen-opt/kernel-gen-opt.cc"],
    visibility = ["//tensorflow/compiler/mlir/tools/kernel_gen/tests:__pkg__"],
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
        "//tensorflow/core:framework",
        "@llvm-project//mlir:mlir_runner_utils",
    ],
)
