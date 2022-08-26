# buildifier: disable=same-origin-load
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)

# buildifier: disable=same-origin-load
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_system_z_available",
    "tf_proto_library",
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Edge TPU compiler needs to use some compiler passes from kernel_gen.
        "//platforms/darwinn/compiler/...",
        "//tensorflow/compiler/...",
        "//tensorflow/core/kernels/mlir_generated/...",
    ],
)

cc_library(
    name = "kernel_creator",
    srcs = ["kernel_creator.cc"],
    hdrs = ["kernel_creator.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:bufferize",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf_no_fallback",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:chlo_legalize_to_hlo_pass",
        "//tensorflow/compiler/xla/mlir_hlo:gpu_passes",
        "//tensorflow/compiler/xla/mlir_hlo:shape_simplification",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexToStandard",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFToGPU",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
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
        "//tensorflow/core:lib",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
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
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
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
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DTENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":compile_cache_item_proto_cc",
        ":kernel_creator",
        ":tf_gpu_runtime_wrappers",
        ":tf_jit_cache",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:strcat",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:mlir_runner_utils",
    ],
)

cc_library(
    name = "tf_jit_cache",
    srcs = ["tf_jit_cache.cc"],
    hdrs = ["tf_jit_cache.h"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//mlir:ExecutionEngine",
    ],
)

cc_library(
    name = "tf_gpu_runtime_wrappers",
    srcs = if_gpu_is_configured([
        "tf_gpu_runtime_wrappers.cc",
    ]),
    hdrs =
        if_gpu_is_configured([
            "tf_gpu_runtime_wrappers.h",
        ]),
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "@llvm-project//mlir:mlir_runner_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform/default/build_config:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/tsl/platform/default/build_config:stream_executor_rocm",
    ]),
)

tf_proto_library(
    name = "compile_cache_item_proto",
    srcs = ["compile_cache_item.proto"],
    cc_api_version = 2,
)
