package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/selectors:convolution_transposed_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:default_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:dw_convolution_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:fully_connected_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:simple_selectors",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/tasks:elementwise",
        "//tensorflow/lite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "//tensorflow/lite/delegates/gpu/metal/kernels",
    ],
)
