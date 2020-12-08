package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "depthwise_conv_plus_1x1_conv",
    srcs = ["depthwise_conv_plus_1x1_conv.cc"],
    hdrs = ["depthwise_conv_plus_1x1_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "fc_fc_add",
    srcs = ["fc_fc_add.cc"],
    hdrs = ["fc_fc_add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "@com_google_absl//absl/memory",
    ],
)
