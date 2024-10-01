load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_xla//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "tf_cc_binary",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:bufferize",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",  # fixdeps: keep
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToStandard",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFToGPU",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:all_passes",  # fixdeps: keep
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:transforms_gpu_passes",
        "@local_xla//xla/mlir_hlo:transforms_passes",
        "@stablehlo//:chlo_ops",
    ],
)

check_deps(
    name = "hlo_to_kernel_check_deps",
    disallowed_deps = ["@local_xla//xla:literal"],
    deps = [":hlo_to_kernel"],
)

tf_cc_binary(
    name = "hlo_to_kernel",
    srcs = ["hlo_to_kernel.cc"],
    visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen/tests/hlo_to_kernel:__pkg__",
        "//tensorflow/core/kernels/mlir_generated:__pkg__",
    ],
    deps = [
        ":kernel_creator",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ] + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
    ]) + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
    ]),
)

tf_cc_binary(
    name = "kernel-gen-opt",
    srcs = ["tools/kernel-gen-opt/kernel-gen-opt.cc"],
    visibility = ["//tensorflow/compiler/mlir/tools/kernel_gen/tests:__subpackages__"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",  # fixdeps: keep
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@stablehlo//:register",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@llvm-project//mlir:mlir_runner_utils",
        "@local_xla//xla/stream_executor",
    ],
)

cc_library(
    name = "tf_jit_cache",
    srcs = ["tf_jit_cache.cc"],
    hdrs = ["tf_jit_cache.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "@local_tsl//tsl/platform:thread_annotations",
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
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:mlir_runner_utils",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/stream_executor",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_xla//xla/stream_executor/cuda:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_xla//xla/stream_executor/rocm:stream_executor_rocm",
    ]),
)

tf_proto_library(
    name = "compile_cache_item_proto",
    srcs = ["compile_cache_item.proto"],
)
