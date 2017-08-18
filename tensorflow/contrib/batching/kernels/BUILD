# Description:
#   Contains kernels for the batching ops.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "batch_kernels",
    srcs = ["batch_kernels.cc"],
    deps = [
        "//tensorflow/contrib/batching:shared_batch_scheduler_hdrs",
        "//tensorflow/contrib/batching/util:periodic_function_dynamic",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:concat_lib_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "//tensorflow/core/kernels:split_lib_hdrs",
    ],
    alwayslink = 1,
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
)
