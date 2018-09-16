licenses(["notice"])  # Apache 2.0
package(
    default_visibility = [
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "xrt_server",
    srcs = [
        "xrt_server.cc",
    ],
    deps = [
        "//tensorflow:grpc++",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/compiler/xrt:xrt_server"
    ],
)
