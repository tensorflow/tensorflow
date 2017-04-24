licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

cc_library(
    name = "xla_local_launch_op",
    srcs = ["xla_local_launch_op.cc"],
    hdrs = ["xla_local_launch_op.h"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:xla_compilation_cache",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_local_runtime_context",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:tensorflow_opensource",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_device_launch_op",
    srcs = ["xla_device_launch_op.cc"],
    hdrs = ["xla_device_launch_op.h"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:xla_compilation_cache",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

cc_library(
    name = "parallel_check_op",
    srcs = ["parallel_check_op.cc"],
    visibility = ["//tensorflow/compiler/jit:friends"],
    deps = [
        "//tensorflow/compiler/jit/legacy_flags:parallel_check_op_flags",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
