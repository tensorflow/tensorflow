# Description:
#   Contains a tool to dump TensorFlow ops which are not supported
#   in TensorFlow HVX runtime.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

tf_cc_binary(
    name = "hvx_ops_support_checker",
    testonly = 1,
    srcs = ["hvx_ops_support_checker_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
        "//tensorflow/tools/graph_transforms:file_utils",
    ],
)
