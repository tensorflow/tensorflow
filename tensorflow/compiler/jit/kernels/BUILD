licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

cc_library(
    name = "xla_launch_op",
    srcs = ["xla_launch_op.cc"],
    hdrs = ["xla_launch_op.h"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:xla_compilation_cache",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/kernels:variable_ops",
    ],
    alwayslink = 1,
)
