# Description:
#   MLIR-GPU-specific components in XLA service implementation.

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
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "inject_errors_pass",
    srcs = ["inject_errors_pass.cc"],
    hdrs = ["inject_errors_pass.h"],
    deps = [
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
)

cc_library(
    name = "mlir_compiler",
    srcs = ["mlir_compiler.cc"],
    hdrs = ["mlir_compiler.h"],
    deps = [
        ":emission_context",
        ":failover_compiler",
        ":kernel_lowering",
        ":lhlo_dialect_emitter",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:compiler",
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
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/gpu:asm_compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_mlir//:GPUDialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMDialect",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TargetNVVMIR",
    ],
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
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
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
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMDialect",
        "@local_config_mlir//:StandardOps",
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
        "//tensorflow/compiler/mlir/xla:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_linalg",
        "//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@local_config_mlir//:AffineDialectRegistration",
        "@local_config_mlir//:GPUDialect",
        "@local_config_mlir//:GPUDialectRegistration",
        "@local_config_mlir//:GPUToNVVMTransforms",
        "@local_config_mlir//:GPUTransforms",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMDialect",
        "@local_config_mlir//:LLVMTransforms",
        "@local_config_mlir//:Linalg",
        "@local_config_mlir//:LinalgDialectRegistration",
        "@local_config_mlir//:LoopDialectRegistration",
        "@local_config_mlir//:LoopOps",
        "@local_config_mlir//:LoopsToGPUPass",
        "@local_config_mlir//:NVVMDialect",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Transforms",
    ],
)

cc_library(
    name = "mlir_irgen_test_base",
    testonly = True,
    srcs = ["mlir_irgen_test_base.cc"],
    hdrs = ["mlir_irgen_test_base.h"],
    deps = [
        ":failover_compiler",
        ":inject_errors_pass",
        ":mlir_compiler",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/tests:codegen_test_base",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:test",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
    ],
)
