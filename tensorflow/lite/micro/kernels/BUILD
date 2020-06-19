load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "cc_library",
    "micro_copts",
)

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "xtensa_hifimini_legacy",
    define_values = {"tflm_build": "xtensa_hifimini"},
)

package_group(
    name = "micro_top_level",
    packages = ["//tensorflow/lite/micro"],
)

# LINT.IfChange(micro_ops)
cc_library(
    name = "micro_ops",
    srcs = [
        "activations.cc",
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
        "split.cc",
        "strided_slice.cc",
        "sub.cc",
        "tanh.cc",
        "unpack.cc",
    ] + select({
        "//conditions:default": [
            "conv.cc",
            "depthwise_conv.cc",
            "fully_connected.cc",
            "quantize.cc",
            "softmax.cc",
            "svdf.cc",
        ],
        ":xtensa_hifimini_legacy": [
            "xtensa_hifimini_legacy/conv.cc",
            "xtensa_hifimini_legacy/depthwise_conv.cc",
            "xtensa_hifimini_legacy/fixedpoint_utils.h",
            "xtensa_hifimini_legacy/fully_connected.cc",
            "xtensa_hifimini_legacy/quantize.cc",
            "xtensa_hifimini_legacy/softmax.cc",
            "xtensa_hifimini_legacy/svdf.cc",
        ],
    }),
    hdrs = ["micro_ops.h"],
    # TODO(b/153609488): enable embedded build once we can properly support it.
    #build_for_embedded = True,
    copts = micro_copts(),
    visibility = [
        # Needed for micro:op_resolvers but visibility can not be
        # finer-grained than a package.
        #":micro_top_level",
        # Currently setting to public for legacy reasons. Can be made more
        # restrictive once cl/314173854 lands.
        "//visibility:public",
    ],
    deps = [
        ":activation_utils",
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
        "//tensorflow/lite/micro:micro_utils",
    ] + select({
        "//conditions:default": [],
        ":xtensa_hifimini_legacy": [
            #"//third_party/xtensa/cstub64s:hifi_mini",
        ],
    }),
)
# LINT.ThenChange(//tensorflow/lite/micro/kernels/BUILD:portable_optimized_micro_ops)

# LINT.IfChange(portable_optimized_micro_ops)
cc_library(
    name = "portable_optimized_micro_ops",
    srcs = [
        "activations.cc",
        "add.cc",
        "arg_min_max.cc",
        "ceil.cc",
        "circular_buffer.cc",
        "comparisons.cc",
        "concatenation.cc",
        "conv.cc",
        "dequantize.cc",
        "elementwise.cc",
        "ethosu.cc",
        "floor.cc",
        "fully_connected.cc",
        "l2norm.cc",
        "logical.cc",
        "logistic.cc",
        "maximum_minimum.cc",
        "mul.cc",
        "neg.cc",
        "pack.cc",
        "pad.cc",
        "pooling.cc",
        "portable_optimized/depthwise_conv.cc",
        "prelu.cc",
        "quantize.cc",
        "reduce.cc",
        "reshape.cc",
        "resize_nearest_neighbor.cc",
        "round.cc",
        "softmax.cc",
        "split.cc",
        "strided_slice.cc",
        "sub.cc",
        "svdf.cc",
        "tanh.cc",
        "unpack.cc",
    ],
    hdrs = ["micro_ops.h"],
    copts = micro_copts(),
    visibility = [
        # Needed for micro:portable_optimized_ops_resolver but visibility can not be
        # finer-grained than a package.
        ":micro_top_level",
    ],
    deps = [
        ":activation_utils",
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
        "//tensorflow/lite/micro:micro_utils",
    ],
)
# LINT.ThenChange(//tensorflow/lite/micro/kernels/BUILD:micro_ops)

test_suite(
    name = "all_tests",
)

tflite_micro_cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "pooling_test",
    srcs = [
        "pooling_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "portable_optimized_depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:portable_optimized_op_resolver",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "fully_connected_test",
    srcs = [
        "fully_connected_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "logistic_test",
    srcs = [
        "logistic_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "svdf_test",
    srcs = [
        "svdf_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "conv_test",
    srcs = [
        "conv_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "prelu_test",
    srcs = [
        "prelu_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "floor_test",
    srcs = [
        "floor_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "logical_test",
    srcs = [
        "logical_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "neg_test",
    srcs = [
        "neg_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "maximum_minimum_test",
    srcs = [
        "maximum_minimum_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "mul_test",
    srcs = [
        "mul_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "sub_test",
    srcs = [
        "sub_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "arg_min_max_test",
    srcs = [
        "arg_min_max_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "comparisons_test",
    srcs = [
        "comparisons_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "ceil_test",
    srcs = [
        "ceil_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "round_test",
    srcs = [
        "round_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "strided_slice_test",
    srcs = [
        "strided_slice_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "pack_test",
    srcs = [
        "pack_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "unpack_test",
    srcs = [
        "unpack_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "split_test",
    srcs = [
        "split_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "add_test",
    srcs = [
        "add_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
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
    build_for_embedded = True,
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "micro_utils",
    hdrs = ["micro_utils.h"],
    build_for_embedded = True,
)

tflite_micro_cc_test(
    name = "reshape_test",
    srcs = [
        "reshape_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "concatenation_test",
    srcs = [
        "concatenation_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "pad_test",
    srcs = [
        "pad_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "circular_buffer_test",
    srcs = [
        "circular_buffer_test.cc",
    ],
    deps = [
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "resize_nearest_neighbor_test",
    srcs = [
        "resize_nearest_neighbor_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "l2norm_test",
    srcs = [
        "l2norm_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
