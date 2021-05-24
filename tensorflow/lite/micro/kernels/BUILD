load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

config_setting(
    name = "xtensa_hifimini",
    define_values = {"tflm_build": "xtensa_hifimini"},
)

package_group(
    name = "micro",
    packages = ["//tensorflow/lite/micro/..."],
)

package_group(
    name = "micro_top_level",
    packages = ["//tensorflow/lite/micro"],
)

####################################
# C++ libraries
####################################

cc_library(
    name = "activation_utils",
    hdrs = ["activation_utils.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
    ],
)

cc_library(
    name = "circular_buffer_flexbuffers_generated_data",
    srcs = [
        "circular_buffer_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "circular_buffer_flexbuffers_generated_data.h",
    ],
)

cc_library(
    name = "conv",
    srcs = [
        "conv_common.cc",
    ] + select({
        "//conditions:default": [
            "conv.cc",
        ],
        ":xtensa_hifimini": [
            "xtensa/conv.cc",
        ],
    }),
    hdrs = ["conv.h"],
    copts = micro_copts(),
    visibility = [
        # Kernel variants need to be visible to the examples and benchmarks.
        ":micro",
    ],
    deps = [
        ":fixedpoint_utils",
        ":kernel_util",
        ":xtensa",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

cc_library(
    name = "conv_test_common",
    srcs = [
        "conv_test_common.cc",
    ],
    hdrs = [
        "conv_test.h",
    ],
    deps = [
        ":kernel_runner",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = [
        "depthwise_conv_common.cc",
    ] + select({
        "//conditions:default": [
            "depthwise_conv.cc",
        ],
        ":xtensa_hifimini": [
            "xtensa/depthwise_conv.cc",
        ],
    }),
    hdrs = ["depthwise_conv.h"],
    copts = micro_copts(),
    visibility = [
        # Kernel variants need to be visible to the examples and benchmarks.
        ":micro",
    ],
    deps = [
        ":conv",
        ":fixedpoint_utils",
        ":kernel_util",
        ":xtensa",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

cc_library(
    name = "ethosu",
    srcs = [
        "ethosu.cc",
    ],
    hdrs = ["ethosu.h"],
    copts = micro_copts(),
    visibility = [
        # Kernel variants need to be visible to the examples and benchmarks.
        ":micro",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "fixedpoint_utils",
    hdrs = select({
        "//conditions:default": [
        ],
        ":xtensa_hifimini": [
            "xtensa/fixedpoint_utils.h",
        ],
    }),
    copts = micro_copts(),
    deps = [
        ":xtensa",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
            "//tensorflow/lite/kernels/internal:compatibility",
        ],
    }),
)

cc_library(
    name = "detection_postprocess_flexbuffers_generated_data",
    srcs = [
        "detection_postprocess_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "detection_postprocess_flexbuffers_generated_data.h",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = [
        "fully_connected_common.cc",
    ] + select({
        "//conditions:default": [
            "fully_connected.cc",
        ],
        ":xtensa_hifimini": [
            "xtensa/fully_connected.cc",
        ],
    }),
    hdrs = ["fully_connected.h"],
    copts = micro_copts(),
    visibility = [
        # Kernel variants need to be visible to the examples and benchmarks.
        ":micro",
    ],
    deps = [
        ":fixedpoint_utils",
        ":kernel_util",
        ":xtensa",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

cc_library(
    name = "kernel_runner",
    srcs = [
        "kernel_runner.cc",
    ],
    hdrs = ["kernel_runner.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:micro_allocator",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:mock_micro_graph",
        "//tensorflow/lite/micro:test_helpers",
    ],
)

cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = ["kernel_util.h"],
    visibility = [
        ":micro",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/micro:debug_log",
    ],
)

cc_library(
    name = "micro_ops",
    srcs = [
        "activations.cc",
        "hard_swish.cc",
        "add.cc",
        "add_n.cc",
        "arg_min_max.cc",
        "batch_to_space_nd.cc",
        "cast.cc",
        "ceil.cc",
        "circular_buffer.cc",
        "comparisons.cc",
        "concatenation.cc",
        "cumsum.cc",
        "depth_to_space.cc",
        "dequantize.cc",
        "detection_postprocess.cc",
        "elementwise.cc",
        "elu.cc",
        "exp.cc",
        "expand_dims.cc",
        "fill.cc",
        "floor.cc",
        "floor_div.cc",
        "floor_mod.cc",
        "gather_nd.cc",
        "if.cc",
        "l2norm.cc",
        "l2_pool_2d.cc",
        "leaky_relu.cc",
        "logical.cc",
        "logistic.cc",
        "log_softmax.cc",
        "maximum_minimum.cc",
        "mul.cc",
        "neg.cc",
        "pack.cc",
        "pad.cc",
        "pooling.cc",
        "prelu.cc",
        "quantize_common.cc",
        "reduce.cc",
        "reshape.cc",
        "resize_bilinear.cc",
        "resize_nearest_neighbor.cc",
        "round.cc",
        "shape.cc",
        "softmax_common.cc",
        "space_to_batch_nd.cc",
        "split.cc",
        "split_v.cc",
        "squeeze.cc",
        "strided_slice.cc",
        "sub.cc",
        "svdf_common.cc",
        "tanh.cc",
        "transpose_conv.cc",
        "transpose.cc",
        "unpack.cc",
        "zeros_like.cc",
    ] + select({
        "//conditions:default": [
            "quantize.cc",
            "softmax.cc",
            "svdf.cc",
        ],
        ":xtensa_hifimini": [
            "xtensa/quantize.cc",
            "xtensa/softmax.cc",
            "xtensa/svdf.cc",
        ],
    }),
    hdrs = [
        "micro_ops.h",
        "quantize.h",
        "softmax.h",
        "svdf.h",
    ],
    copts = micro_copts(),
    visibility = [
        # Public visibility to allow application code to select kernel variants.
        "//visibility:public",
    ],
    deps = [
        ":activation_utils",
        ":fixedpoint_utils",
        ":kernel_util",
        ":micro_utils",
        ":xtensa",
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/micro:micro_graph",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

cc_library(
    name = "micro_utils",
    hdrs = ["micro_utils.h"],
)

cc_library(
    name = "xtensa",
    hdrs = select({
        "//conditions:default": [
        ],
        ":xtensa_hifimini": [
            "xtensa/xtensa.h",
        ],
    }),
    copts = micro_copts(),
    deps = select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

####################################
# C++ tests
####################################

cc_test(
    name = "activations_test",
    srcs = [
        "activations_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "add_n_test",
    srcs = [
        "add_n_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "add_test",
    srcs = [
        "add_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "arg_min_max_test",
    srcs = [
        "arg_min_max_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "batch_to_space_nd_test",
    srcs = [
        "batch_to_space_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "cast_test",
    srcs = ["cast_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "ceil_test",
    srcs = [
        "ceil_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "circular_buffer_test",
    srcs = [
        "circular_buffer_test.cc",
    ],
    deps = [
        "circular_buffer_flexbuffers_generated_data",
        ":kernel_runner",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "comparisons_test",
    srcs = [
        "comparisons_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "concatenation_test",
    srcs = [
        "concatenation_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "conv_test",
    srcs = [
        "conv_test.cc",
    ],
    deps = [
        ":conv_test_common",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "cumsum_test",
    srcs = [
        "cumsum_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "depth_to_space_test",
    srcs = [
        "depth_to_space_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "dequantize_test",
    srcs = [
        "dequantize_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "detection_postprocess_test",
    srcs = [
        "detection_postprocess_test.cc",
    ],
    deps = [
        ":detection_postprocess_flexbuffers_generated_data",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "elu_test",
    srcs = [
        "elu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "exp_test",
    srcs = ["exp_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "expand_dims_test",
    srcs = ["expand_dims_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "fill_test",
    srcs = [
        "fill_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "floor_div_test",
    srcs = ["floor_div_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "floor_mod_test",
    srcs = ["floor_mod_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "floor_test",
    srcs = [
        "floor_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = [
        "fully_connected_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "gather_nd_test",
    srcs = [
        "gather_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "hard_swish_test",
    srcs = ["hard_swish_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "if_test",
    srcs = ["if_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:mock_micro_graph",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "l2norm_test",
    srcs = [
        "l2norm_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "l2_pool_2d_test",
    srcs = [
        "l2_pool_2d_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "leaky_relu_test",
    srcs = [
        "leaky_relu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "logical_test",
    srcs = [
        "logical_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "logistic_test",
    srcs = [
        "logistic_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "log_softmax_test",
    srcs = [
        "log_softmax_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "maximum_minimum_test",
    srcs = [
        "maximum_minimum_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "mul_test",
    srcs = [
        "mul_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "neg_test",
    srcs = [
        "neg_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "pack_test",
    srcs = [
        "pack_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "pad_test",
    srcs = [
        "pad_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",  # TODO(b/175133159): currently failing with asan and msan
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = [
        "pooling_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = [
        "prelu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = [
        "quantization_util_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "quantize_test",
    srcs = [
        "quantize_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "reduce_test",
    srcs = [
        "reduce_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = [
        "reshape_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = [
        "resize_bilinear_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "resize_nearest_neighbor_test",
    srcs = [
        "resize_nearest_neighbor_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "round_test",
    srcs = [
        "round_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = [
        "softmax_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "space_to_batch_nd_test",
    srcs = [
        "space_to_batch_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "split_test",
    srcs = [
        "split_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "split_v_test",
    srcs = [
        "split_v_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "squeeze_test",
    srcs = ["squeeze_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "strided_slice_test",
    srcs = [
        "strided_slice_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "sub_test",
    srcs = [
        "sub_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "svdf_test",
    srcs = [
        "svdf_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "transpose_conv_test",
    srcs = [
        "transpose_conv_test.cc",
    ],
    deps = [
        ":conv_test_common",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "unpack_test",
    srcs = [
        "unpack_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "zeros_like_test",
    srcs = ["zeros_like_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
