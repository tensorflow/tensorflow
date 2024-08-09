load("//tensorflow:tensorflow.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

_selectors_package = (
    "//tensorflow/lite/delegates/gpu/common/selectors/" +
    if_google("google", "default")
)

cc_library(
    name = "convolution_selector",
    hdrs = ["convolution_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        _selectors_package + ":convolution_selector",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    hdrs = ["convolution_transposed_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "@com_google_absl//absl/memory",
        _selectors_package + ":convolution_transposed_selector",
    ],
)

cc_library(
    name = "default_selector",
    hdrs = ["default_selector.h"],
    deps = [
        ":subgraph",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        _selectors_package + ":default_selector",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    hdrs = ["dw_convolution_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":dw_convolution_selector",
    ],
)

cc_library(
    name = "fully_connected_selector",
    hdrs = ["fully_connected_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":fully_connected_selector",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":convolution_selector",
        ":convolution_transposed_selector",
        ":default_selector",
        ":dw_convolution_selector",
        ":fully_connected_selector",
        ":simple_selectors",
        ":subgraph",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:flops_util",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:weights_conversion",
        "//tensorflow/lite/delegates/gpu/common/task:weights_layout",
        "//tensorflow/lite/delegates/gpu/common/tasks:elementwise",
        "//tensorflow/lite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "simple_selectors",
    srcs = ["simple_selectors.cc"],
    hdrs = ["simple_selectors.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/tasks:add",
        "//tensorflow/lite/delegates/gpu/common/tasks:cast",
        "//tensorflow/lite/delegates/gpu/common/tasks:concat_xy",
        "//tensorflow/lite/delegates/gpu/common/tasks:concat_z",
        "//tensorflow/lite/delegates/gpu/common/tasks:cumsum",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv",
        "//tensorflow/lite/delegates/gpu/common/tasks:gather",
        "//tensorflow/lite/delegates/gpu/common/tasks:lstm",
        "//tensorflow/lite/delegates/gpu/common/tasks:max_unpooling",
        "//tensorflow/lite/delegates/gpu/common/tasks:one_hot",
        "//tensorflow/lite/delegates/gpu/common/tasks:padding",
        "//tensorflow/lite/delegates/gpu/common/tasks:pooling",
        "//tensorflow/lite/delegates/gpu/common/tasks:prelu",
        "//tensorflow/lite/delegates/gpu/common/tasks:quantize_and_dequantize",
        "//tensorflow/lite/delegates/gpu/common/tasks:reduce",
        "//tensorflow/lite/delegates/gpu/common/tasks:relu",
        "//tensorflow/lite/delegates/gpu/common/tasks:resampler",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshape",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshapex4",
        "//tensorflow/lite/delegates/gpu/common/tasks:resize",
        "//tensorflow/lite/delegates/gpu/common/tasks:select_v2",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax1x1",
        "//tensorflow/lite/delegates/gpu/common/tasks:space_to_depth",
        "//tensorflow/lite/delegates/gpu/common/tasks:split",
        "//tensorflow/lite/delegates/gpu/common/tasks:strided_slice",
        "//tensorflow/lite/delegates/gpu/common/tasks:tile",
        "//tensorflow/lite/delegates/gpu/common/tasks:transpose",
        "//tensorflow/lite/delegates/gpu/common/tasks:winograd",
    ],
)

cc_library(
    name = "special_selector",
    srcs = ["special_selector.cc"],
    hdrs = ["special_selector.h"],
    deps = [
        ":subgraph",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "//tensorflow/lite/delegates/gpu/common/tasks/special:conv_pointwise",
        "//tensorflow/lite/delegates/gpu/common/tasks/special:dw7x7_conv2to6_concat_conv8to8",
        "//tensorflow/lite/delegates/gpu/common/tasks/special:fc_fc_add",
        "//tensorflow/lite/delegates/gpu/common/tasks/special:thin_pointwise_fuser",
    ],
)

cc_library(
    name = "subgraph",
    srcs = ["subgraph.cc"],
    hdrs = ["subgraph.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
    ],
)
