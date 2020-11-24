load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

licenses(["notice"])  # Apache 2.0

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
    deps = select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
            "//tensorflow/lite/kernels/internal:compatibility",
        ],
    }),
)

cc_library(
    name = "fully_connected",
    srcs = select({
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
    name = "micro_ops",
    srcs = [
        "activations.cc",
        "hard_swish.cc",
        "add.cc",
        "arg_min_max.cc",
        "ceil.cc",
        "circular_buffer.cc",
        "comparisons.cc",
        "concatenation.cc",
        "dequantize.cc",
        "elementwise.cc",
        "ethosu.cc",
        "floor.cc",
        "l2norm.cc",
        "logical.cc",
        "logistic.cc",
        "maximum_minimum.cc",
        "mul.cc",
        "neg.cc",
        "pack.cc",
        "pad.cc",
        "pooling.cc",
        "prelu.cc",
        "reduce.cc",
        "reshape.cc",
        "resize_nearest_neighbor.cc",
        "round.cc",
        "shape.cc",
        "split.cc",
        "split_v.cc",
        "squeeze.cc",
        "strided_slice.cc",
        "sub.cc",
        "tanh.cc",
        "unpack.cc",
    ] + select({
        "//conditions:default": [
            "conv.cc",
            "depthwise_conv.cc",
            "quantize.cc",
            "softmax.cc",
            "svdf.cc",
        ],
        ":xtensa_hifimini": [
            "xtensa/conv.cc",
            "xtensa/depthwise_conv.cc",
            "xtensa/quantize.cc",
            "xtensa/softmax.cc",
            "xtensa/svdf.cc",
        ],
    }),
    hdrs = ["micro_ops.h"],
    copts = micro_copts(),
    visibility = [
        # Needed for micro:op_resolvers but visibility can not be finer-grained
        # than a package.
        ":micro_top_level",
    ],
    deps = [
        ":activation_utils",
        ":kernel_util",
        ":fixedpoint_utils",
        ":micro_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_utils",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)

test_suite(
    name = "all_tests",
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
    name = "conv_test",
    srcs = [
        "conv_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

cc_library(
    name = "activation_utils",
    hdrs = ["activation_utils.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
    ],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

cc_library(
    name = "kernel_runner",
    srcs = [
        "kernel_runner.cc",
    ],
    hdrs = ["kernel_runner.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = ["kernel_util.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "micro_utils",
    hdrs = ["micro_utils.h"],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
    name = "pad_test",
    srcs = [
        "pad_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
    name = "circular_buffer_test",
    srcs = [
        "circular_buffer_test.cc",
    ],
    deps = [
        ":kernel_runner",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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

tflite_micro_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
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

tflite_micro_cc_test(
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