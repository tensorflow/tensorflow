# Description:
#   MLIR-GPU-specific components in XLA service implementation.

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
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
    name = "mlir_compiler",
    srcs = ["mlir_compiler.cc"],
    hdrs = ["mlir_compiler.h"],
    deps = [
        ":failover_compiler",
        ":lhlo_dialect_emitter",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:gpu_constants",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_hlo_schedule",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler_impl",
        "//tensorflow/compiler/xla/service/gpu:stream_assignment",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:stream_executor_headers",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMDialect",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "lhlo_dialect_emitter",
    srcs = ["lhlo_dialect_emitter.cc"],
    hdrs = ["lhlo_dialect_emitter.h"],
    deps = [
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
