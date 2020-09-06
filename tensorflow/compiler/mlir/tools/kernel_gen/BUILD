load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = ["//tensorflow/compiler/mlir/..."],
)

cc_library(
    name = "gpu_binary_creator",
    srcs = ["gpu_binary_creator.cc"],
    hdrs = ["gpu_binary_creator.h"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TargetNVVMIR",
        "@llvm-project//mlir:TargetROCDLIR",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/hlo:materialize_broadcasts",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/hlo:unfuse_batch_norm",  # buildcleaner: keep
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/mlir_gpu:kernel_lowering",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/core:lib",
    ] + if_cuda([
        "//tensorflow/stream_executor/gpu:asm_compiler",
    ]) + if_rocm([
        "//tensorflow/core/platform:rocm_rocdl_path",
    ]),
)

tf_cc_binary(
    name = "tf_to_gpu_binary",
    srcs = ["tf_to_gpu_binary.cc"],
    visibility = ["//tensorflow/core/kernels/mlir_generated:__pkg__"],
    deps = [
        ":gpu_binary_creator",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
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
