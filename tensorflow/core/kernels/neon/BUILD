# Description:
# Kernel implementations using Neon intrinsics.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)

tf_kernel_library(
    name = "neon_depthwise_conv_op",
    hdrs = [
        "depthwiseconv_float.h",
        "types.h",
    ],
    features = ["-parse_headers"],  # included gemmlowp headers are not self-contained
    prefix = "neon_depthwise_conv_op",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:ops_util",
        "@gemmlowp",
    ],
)
