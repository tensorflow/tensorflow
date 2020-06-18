# Contains graph rewrites for TPU runtimes and optimizations.

package(
    default_visibility = [
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/stream_executor/tpu:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "distributed_tpu_configuration_rewrite_registration",
    srcs = ["distributed_tpu_configuration_rewrite_registration.cc"],
    deps = [
        ":distributed_tpu_configuration_rewrite_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "distributed_tpu_configuration_rewrite_pass",
    srcs = [
        "distributed_tpu_configuration_rewrite_pass.cc",
    ],
    hdrs = [
        "distributed_tpu_configuration_rewrite_pass.h",
    ],
    deps = [
        ":distributed_tpu_rewrite_helpers",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_init_mode",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_options",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_helpers",
    srcs = ["distributed_tpu_rewrite_helpers.cc"],
    hdrs = ["distributed_tpu_rewrite_helpers.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tpu:tpu_defs",
    ],
)
