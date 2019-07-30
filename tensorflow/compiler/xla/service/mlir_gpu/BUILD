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
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service/gpu:gpu_constants",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/core:lib",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMDialect",
    ],
    alwayslink = True,  # Contains compiler registration
)
