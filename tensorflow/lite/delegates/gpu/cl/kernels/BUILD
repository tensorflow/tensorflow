load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//tensorflow/lite/delegates/gpu:build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    # TODO(b/279977471) Once b/279347631 is resolved, check for heap again
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:add_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "cast_test",
    srcs = ["cast_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:cast_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_library(
    name = "cl_test",
    testonly = 1,
    srcs = ["cl_test.cc"],
    hdrs = ["cl_test.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:cl_operation",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:testing_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:concat_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "conv_constants_test",
    srcs = ["conv_constants_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_constants_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "conv_generic_test",
    srcs = ["conv_generic_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_generic_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "conv_weights_converter_test",
    srcs = ["conv_weights_converter_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:conv_weights_converter_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu:spi",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_arguments",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_errors",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type_util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:arguments",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
        "//tensorflow/lite/delegates/gpu/common/tasks:conversion",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "convolution_transposed_test",
    srcs = ["convolution_transposed_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "convolution_transposed_3x3_test",
    srcs = ["convolution_transposed_3x3_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_3x3_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "convolution_transposed_3x3_thin_test",
    srcs = ["convolution_transposed_3x3_thin_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_3x3_thin_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "convolution_transposed_4x4_test",
    srcs = ["convolution_transposed_4x4_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_4x4_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "convolution_transposed_thin_test",
    srcs = ["convolution_transposed_thin_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:convolution_transposed_thin_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "cumsum_test",
    srcs = ["cumsum_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:cumsum_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "depthwise_conv_3x3_test",
    srcs = ["depthwise_conv_3x3_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv_3x3_stride_h2_test_util",
        "//tensorflow/lite/delegates/gpu/common/tasks:depthwise_conv_3x3_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:elementwise_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/tasks:fully_connected",
        "//tensorflow/lite/delegates/gpu/common/tasks:fully_connected_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "gather_test",
    srcs = ["gather_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:gather_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "lstm_test",
    srcs = ["lstm_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:lstm_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "lstm_full_test",
    srcs = ["lstm_full_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "max_unpooling_test",
    srcs = ["max_unpooling_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:max_unpooling_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "mean_stddev_normalization_test",
    srcs = ["mean_stddev_normalization_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:mean_stddev_normalization_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "one_hot_test",
    srcs = ["one_hot_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:one_hot_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:padding_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:pooling_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:prelu_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "quantize_and_dequantize_test",
    srcs = ["quantize_and_dequantize_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:quantize_and_dequantize_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:reduce_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:relu_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl/kernels:cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:resampler_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshape_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "reshapex4_test",
    srcs = ["reshapex4_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:reshape_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "select_v2_test",
    srcs = ["select_v2_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:select_v2_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "softmax1x1_test",
    srcs = ["softmax1x1_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:softmax_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:space_to_depth_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "split_test",
    srcs = ["split_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:split_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "strided_slice_test",
    srcs = ["strided_slice_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:strided_slice_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "tile_test",
    srcs = ["tile_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:tile_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:transpose_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:resize_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_test(
    name = "winograd_test",
    srcs = ["winograd_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/tasks:winograd_test_util",
    ] + gtest_main_no_heapcheck_deps(),
)

test_suite(
    name = "all_tests",
    tests = [
        "add_test",
        "cast_test",
        "concat_test",
        "conv_constants_test",
        "conv_generic_test",
        "conv_weights_converter_test",
        "convolution_transposed_3x3_test",
        "convolution_transposed_3x3_thin_test",
        "convolution_transposed_4x4_test",
        "convolution_transposed_test",
        "convolution_transposed_thin_test",
        "depthwise_conv_3x3_test",
        "depthwise_conv_test",
        "elementwise_test",
        "fully_connected_test",
        "gather_test",
        "lstm_test",
        "max_unpooling_test",
        "mean_stddev_normalization_test",
        "padding_test",
        "pooling_test",
        "prelu_test",
        "reduce_test",
        "relu_test",
        "reshape_test",
        "reshapex4_test",
        "resize_test",
        "softmax1x1_test",
        "softmax_test",
        "split_test",
        "strided_slice_test",
        "tile_test",
        "transpose_test",
        "winograd_test",
    ],
)
