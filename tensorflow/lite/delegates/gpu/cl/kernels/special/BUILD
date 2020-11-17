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

cc_library(
    name = "fc_fc_add",
    srcs = ["fc_fc_add.cc"],
    hdrs = ["fc_fc_add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:device_info",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/kernels:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/memory",
    ],
)
