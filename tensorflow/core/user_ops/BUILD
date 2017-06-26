load("//tensorflow:tensorflow.bzl", "tf_custom_op_library", "cc_header_only_library")

cc_header_only_library(
    name = "conv_2d",
    deps = [
        "//tensorflow/core/kernels:conv_2d",
    ],
)

cc_header_only_library(
    name = "pooling_ops_hdrs",
    deps = [
        "//tensorflow/core/kernels:pooling_ops_hdrs",
    ],
)

cc_header_only_library(
    name = "cast_ops_hdrs",
    deps = [
        "//tensorflow/core/kernels:cast_op",
    ],
)

tf_custom_op_library(
    name = "bnns.so",
    srcs = ["bnns_conv.cc", "bnns_maxpool.cc"],
    deps = [
        ":conv_2d",
        ":pooling_ops_hdrs",
        ":cast_ops_hdrs",
    ],
)
