package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "convolution_selector",
    srcs = ["convolution_selector.cc"],
    hdrs = ["convolution_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:model_hints",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_buffer",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_constants",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_powervr",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_texture",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/kernels:work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    srcs = ["convolution_transposed_selector.cc"],
    hdrs = ["convolution_transposed_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl/kernels:convolution_transposed",
        "//tensorflow/lite/delegates/gpu/cl/kernels:convolution_transposed_3x3_thin",
        "//tensorflow/lite/delegates/gpu/cl/kernels:convolution_transposed_thin",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    srcs = ["dw_convolution_selector.cc"],
    hdrs = ["dw_convolution_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl/kernels:depth_wise_conv",
        "//tensorflow/lite/delegates/gpu/cl/kernels:depth_wise_conv_3x3",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "fully_connected_selector",
    srcs = ["fully_connected_selector.cc"],
    hdrs = ["fully_connected_selector.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_powervr",
        "//tensorflow/lite/delegates/gpu/cl/kernels:conv_texture",
        "//tensorflow/lite/delegates/gpu/cl/kernels:fully_connected_texture",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":convolution_selector",
        ":convolution_transposed_selector",
        ":dw_convolution_selector",
        ":fully_connected_selector",
        ":simple_selectors",
        "//tensorflow/lite/delegates/gpu/cl:model_hints",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl/kernels:elementwise",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "simple_selectors",
    srcs = ["simple_selectors.cc"],
    hdrs = ["simple_selectors.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl/kernels:add",
        "//tensorflow/lite/delegates/gpu/cl/kernels:apply_mask",
        "//tensorflow/lite/delegates/gpu/cl/kernels:concat_xy",
        "//tensorflow/lite/delegates/gpu/cl/kernels:concat_z",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/kernels:lstm",
        "//tensorflow/lite/delegates/gpu/cl/kernels:max_unpooling",
        "//tensorflow/lite/delegates/gpu/cl/kernels:multiply_add",
        "//tensorflow/lite/delegates/gpu/cl/kernels:padding",
        "//tensorflow/lite/delegates/gpu/cl/kernels:pooling",
        "//tensorflow/lite/delegates/gpu/cl/kernels:prelu",
        "//tensorflow/lite/delegates/gpu/cl/kernels:relu",
        "//tensorflow/lite/delegates/gpu/cl/kernels:reshape",
        "//tensorflow/lite/delegates/gpu/cl/kernels:reshapex4",
        "//tensorflow/lite/delegates/gpu/cl/kernels:softmax",
        "//tensorflow/lite/delegates/gpu/cl/kernels:softmax1x1",
        "//tensorflow/lite/delegates/gpu/cl/kernels:strided_slice",
        "//tensorflow/lite/delegates/gpu/cl/kernels:transpose",
        "//tensorflow/lite/delegates/gpu/cl/kernels:upsample",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)
