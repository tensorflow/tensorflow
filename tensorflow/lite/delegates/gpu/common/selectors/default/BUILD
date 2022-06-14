package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convolution_selector",
    srcs = ["convolution_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_constants",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_generic",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_metal",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_metal_simd",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_weights_converter",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    srcs = ["convolution_transposed_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_3x3",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_3x3_thin",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_4x4",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_thin",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "default_selector",
    srcs = ["default_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    srcs = ["dw_convolution_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv_3x3",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv_3x3_stride_h2",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "fully_connected_selector",
    srcs = ["fully_connected_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_generic",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_metal",
        "//tensorflow/lite/delegates/gpu/common/tasks:fully_connected",
    ],
)
