# Description:
# Kernel implementations using Neon intrinsics.
#
package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)

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

tf_kernel_library(
    name = "neon_depthwise_conv_op",
    hdrs = [
        "depthwiseconv_float.h",
        "types.h",
    ],
    prefix = "neon_depthwise_conv_op",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/kernels:ops_util",
        "@gemmlowp//:gemmlowp",
    ],
)
