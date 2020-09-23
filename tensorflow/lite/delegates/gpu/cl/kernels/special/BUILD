package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "depthwise_conv_plus_1x1_conv",
    srcs = ["depthwise_conv_plus_1x1_conv.cc"],
    hdrs = ["depthwise_conv_plus_1x1_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:gpu_object",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/kernels:util",
        "//tensorflow/lite/delegates/gpu/cl/kernels:work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)
