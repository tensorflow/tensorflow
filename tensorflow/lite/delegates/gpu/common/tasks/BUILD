package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "add_test_util",
    testonly = 1,
    srcs = ["add_test_util.cc"],
    hdrs = ["add_test_util.h"],
    deps = [
        ":add",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "concat_xy",
    srcs = ["concat_xy.cc"],
    hdrs = ["concat_xy.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "concat_z",
    srcs = ["concat_z.cc"],
    hdrs = ["concat_z.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "concat_test_util",
    testonly = 1,
    srcs = ["concat_test_util.cc"],
    hdrs = ["concat_test_util.h"],
    deps = [
        ":concat_xy",
        ":concat_z",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_buffer_1x1",
    srcs = ["conv_buffer_1x1.cc"],
    hdrs = ["conv_buffer_1x1.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "conv_buffer_1x1_test_util",
    testonly = 1,
    srcs = ["conv_buffer_1x1_test_util.cc"],
    hdrs = ["conv_buffer_1x1_test_util.h"],
    deps = [
        ":conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_constants",
    srcs = ["conv_constants.cc"],
    hdrs = ["conv_constants.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_constants_test_util",
    testonly = 1,
    srcs = ["conv_constants_test_util.cc"],
    hdrs = ["conv_constants_test_util.h"],
    deps = [
        ":conv_constants",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_metal",
    srcs = ["conv_metal.cc"],
    hdrs = ["conv_metal.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_powervr",
    srcs = ["conv_powervr.cc"],
    hdrs = ["conv_powervr.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_powervr_test_util",
    testonly = 1,
    srcs = ["conv_powervr_test_util.cc"],
    hdrs = ["conv_powervr_test_util.h"],
    deps = [
        ":conv_powervr",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_weights_converter",
    srcs = ["conv_weights_converter.cc"],
    hdrs = ["conv_weights_converter.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "conv_weights_converter_test_util",
    testonly = 1,
    srcs = ["conv_weights_converter_test_util.cc"],
    hdrs = ["conv_weights_converter_test_util.h"],
    deps = [
        ":conv_weights_converter",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
    ],
)

cc_library(
    name = "convolution_transposed",
    srcs = ["convolution_transposed.cc"],
    hdrs = ["convolution_transposed.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_transposed_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_test_util.cc"],
    hdrs = ["convolution_transposed_test_util.h"],
    deps = [
        ":convolution_transposed",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_3x3",
    srcs = ["convolution_transposed_3x3.cc"],
    hdrs = ["convolution_transposed_3x3.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_3x3_test_util.cc"],
    hdrs = ["convolution_transposed_3x3_test_util.h"],
    deps = [
        ":convolution_transposed_3x3",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin",
    srcs = ["convolution_transposed_3x3_thin.cc"],
    hdrs = ["convolution_transposed_3x3_thin.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_3x3_thin_test_util.cc"],
    hdrs = ["convolution_transposed_3x3_thin_test_util.h"],
    deps = [
        ":convolution_transposed_3x3_thin",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_4x4",
    srcs = ["convolution_transposed_4x4.cc"],
    hdrs = ["convolution_transposed_4x4.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_4x4_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_4x4_test_util.cc"],
    hdrs = ["convolution_transposed_4x4_test_util.h"],
    deps = [
        ":convolution_transposed_4x4",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_thin",
    srcs = ["convolution_transposed_thin.cc"],
    hdrs = ["convolution_transposed_thin.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_thin_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_thin_test_util.cc"],
    hdrs = ["convolution_transposed_thin_test_util.h"],
    deps = [
        ":convolution_transposed_thin",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "depthwise_conv_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_test_util.cc"],
    hdrs = ["depthwise_conv_test_util.h"],
    deps = [
        ":depthwise_conv",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "depthwise_conv_3x3",
    srcs = ["depthwise_conv_3x3.cc"],
    hdrs = ["depthwise_conv_3x3.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_3x3_test_util.cc"],
    hdrs = ["depthwise_conv_3x3_test_util.h"],
    deps = [
        ":depthwise_conv_3x3",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_stride_h2",
    srcs = ["depthwise_conv_3x3_stride_h2.cc"],
    hdrs = ["depthwise_conv_3x3_stride_h2.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_stride_h2_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_3x3_stride_h2_test_util.cc"],
    hdrs = ["depthwise_conv_3x3_stride_h2_test_util.h"],
    deps = [
        ":depthwise_conv_3x3_stride_h2",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "elementwise_test_util",
    testonly = 1,
    srcs = ["elementwise_test_util.cc"],
    hdrs = ["elementwise_test_util.h"],
    deps = [
        ":elementwise",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "fully_connected_test_util",
    testonly = 1,
    srcs = ["fully_connected_test_util.cc"],
    hdrs = ["fully_connected_test_util.h"],
    deps = [
        ":fully_connected",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "lstm_test_util",
    testonly = 1,
    srcs = ["lstm_test_util.cc"],
    hdrs = ["lstm_test_util.h"],
    deps = [
        ":lstm",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "max_unpooling_test_util",
    testonly = 1,
    srcs = ["max_unpooling_test_util.cc"],
    hdrs = ["max_unpooling_test_util.h"],
    deps = [
        ":max_unpooling",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "mean_stddev_normalization",
    srcs = ["mean_stddev_normalization.cc"],
    hdrs = ["mean_stddev_normalization.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "mean_stddev_normalization_test_util",
    testonly = 1,
    srcs = ["mean_stddev_normalization_test_util.cc"],
    hdrs = ["mean_stddev_normalization_test_util.h"],
    deps = [
        ":mean_stddev_normalization",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "padding_test_util",
    testonly = 1,
    srcs = ["padding_test_util.cc"],
    hdrs = ["padding_test_util.h"],
    deps = [
        ":padding",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "pooling_test_util",
    testonly = 1,
    srcs = ["pooling_test_util.cc"],
    hdrs = ["pooling_test_util.h"],
    deps = [
        ":pooling",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "prelu_test_util",
    testonly = 1,
    srcs = ["prelu_test_util.cc"],
    hdrs = ["prelu_test_util.h"],
    deps = [
        ":prelu",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "quantize_and_dequantize_test_util",
    testonly = 1,
    srcs = ["quantize_and_dequantize_test_util.cc"],
    hdrs = ["quantize_and_dequantize_test_util.h"],
    deps = [
        ":quantize_and_dequantize",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
        "//tensorflow/lite/kernels/internal:quantization_util",
    ],
)

cc_library(
    name = "reduce",
    srcs = ["reduce.cc"],
    hdrs = ["reduce.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:kernel_info",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "reduce_test_util",
    testonly = 1,
    srcs = ["reduce_test_util.cc"],
    hdrs = ["reduce_test_util.h"],
    deps = [
        ":reduce",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "relu_test_util",
    testonly = 1,
    srcs = ["relu_test_util.cc"],
    hdrs = ["relu_test_util.h"],
    deps = [
        ":relu",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "reshapex4",
    srcs = ["reshapex4.cc"],
    hdrs = ["reshapex4.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "reshape_test_util",
    testonly = 1,
    srcs = ["reshape_test_util.cc"],
    hdrs = ["reshape_test_util.h"],
    deps = [
        ":reshape",
        ":reshapex4",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "resize_test_util",
    testonly = 1,
    srcs = ["resize_test_util.cc"],
    hdrs = ["resize_test_util.h"],
    deps = [
        ":resize",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "softmax1x1",
    srcs = ["softmax1x1.cc"],
    hdrs = ["softmax1x1.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
    ],
)

cc_library(
    name = "softmax_test_util",
    testonly = 1,
    srcs = ["softmax_test_util.cc"],
    hdrs = ["softmax_test_util.h"],
    deps = [
        ":softmax",
        ":softmax1x1",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "space_to_depth_test_util",
    testonly = 1,
    srcs = ["space_to_depth_test_util.cc"],
    hdrs = ["space_to_depth_test_util.h"],
    deps = [
        ":space_to_depth",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "split_test_util",
    testonly = 1,
    srcs = ["split_test_util.cc"],
    hdrs = ["split_test_util.h"],
    deps = [
        ":split",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "strided_slice",
    srcs = ["strided_slice.cc"],
    hdrs = ["strided_slice.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "strided_slice_test_util",
    testonly = 1,
    srcs = ["strided_slice_test_util.cc"],
    hdrs = ["strided_slice_test_util.h"],
    deps = [
        ":strided_slice",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "transpose",
    srcs = ["transpose.cc"],
    hdrs = ["transpose.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transpose_test_util",
    testonly = 1,
    srcs = ["transpose_test_util.cc"],
    hdrs = ["transpose_test_util.h"],
    deps = [
        ":transpose",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "winograd",
    srcs = ["winograd.cc"],
    hdrs = ["winograd.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "winograd_test_util",
    testonly = 1,
    srcs = ["winograd_test_util.cc"],
    hdrs = ["winograd_test_util.h"],
    deps = [
        ":winograd",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
    ],
)
