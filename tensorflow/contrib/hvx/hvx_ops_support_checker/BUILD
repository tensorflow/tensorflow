# Description:
#   Contains a tool to dump TensorFlow ops which are not supported
#   in TensorFlow HVX runtime.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

cc_binary(
    name = "hvx_ops_support_checker",
    testonly = 1,
    srcs = ["hvx_ops_support_checker_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ],
)
