package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "conv_pointwise",
    srcs = ["conv_pointwise.cc"],
    hdrs = ["conv_pointwise.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw7x7_conv2to6_concat_conv8to8",
    srcs = ["dw7x7_conv2to6_concat_conv8to8.cc"],
    hdrs = ["dw7x7_conv2to6_concat_conv8to8.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:flops_util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:qcom_thin_filter_desc",
    ],
)

cc_library(
    name = "fc_fc_add",
    srcs = ["fc_fc_add.cc"],
    hdrs = ["fc_fc_add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "thin_pointwise_fuser",
    srcs = ["thin_pointwise_fuser.cc"],
    hdrs = ["thin_pointwise_fuser.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:flops_util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/tasks:elementwise",
        "//tensorflow/lite/delegates/gpu/common/tasks:prelu",
        "//tensorflow/lite/delegates/gpu/common/tasks:relu",
        "@com_google_absl//absl/strings",
    ],
)
