load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# Kernel implementations for KDNN-backed ops. Each kernel here is a
# drop-in replacement for the corresponding public op (e.g. _KdnnSigmoid
# replaces Sigmoid). The Grappler remapper substitutes them in when
# IsKDNNEnabled() returns true AND the dtype is supported.

# Shared template for unary element-wise activations. Add new
# activation op instantiations by adding a `KdnnUnaryOp<KDNN::XXX, T>`
# registration in the corresponding .cc file.
cc_library(
    name = "kdnn_unary_op",
    hdrs = ["kdnn_unary_op.h"],
    deps = [
        "//third_party/KDNN:kdnn",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:cpu_op",
        "//tensorflow/core/kernels:ops_util",
    ],
)

cc_library(
    name = "kdnn_sigmoid_op",
    srcs = ["kdnn_sigmoid_op.cc"],
    hdrs = ["kdnn_unary_op.h"],
    deps = [
        ":kdnn_unary_op",
        "//third_party/KDNN:kdnn",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:cpu_op",
        "//tensorflow/core/util:port",
    ],
)
