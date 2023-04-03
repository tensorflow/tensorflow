load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu:spi",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:command_queue",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:gl_program",
        "//tensorflow/lite/delegates/gpu/gl:gl_shader",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "converter_test",
    size = "small",
    srcs = ["converter_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":converter",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":add",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":concat",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conv_test",
    srcs = ["conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":conv",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "custom_registry",
    srcs = ["custom_registry.cc"],
    hdrs = ["custom_registry.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":depthwise_conv",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":elementwise",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":fully_connected",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
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
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "lstm_test",
    srcs = ["lstm_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":lstm",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
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
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "max_unpooling_test",
    srcs = ["max_unpooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":max_unpooling",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mean",
    srcs = ["mean.cc"],
    hdrs = ["mean.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "mean_test",
    srcs = ["mean_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":mean",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mul",
    srcs = ["mul.cc"],
    hdrs = ["mul.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "mul_test",
    srcs = ["mul_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":mul",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "pad",
    srcs = ["pad.cc"],
    hdrs = ["pad.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":pad",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
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
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":pooling",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":prelu",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "quantize_and_dequantize_test",
    srcs = ["quantize_and_dequantize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_and_dequantize",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":relu",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":resampler",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":reshape",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
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
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":resize",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":slice",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":softmax",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":space_to_depth",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/gl:api",
        "//tensorflow/lite/delegates/gpu/gl:compiler_options",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:object_manager",
        "//tensorflow/lite/delegates/gpu/gl:request_gpu_info",
        "//tensorflow/lite/delegates/gpu/gl:runtime_options",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:default_calculator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tile",
    srcs = ["tile.cc"],
    hdrs = ["tile.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "tile_test",
    srcs = ["tile_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_util",
        ":tile",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "transpose_conv",
    srcs = ["transpose_conv.cc"],
    hdrs = ["transpose_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "transpose_conv_test",
    srcs = ["transpose_conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_util",
        ":transpose_conv",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
    ],
)

TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "add",
    "concat",
    "conv",
    "depthwise_conv",
    "elementwise",
    "fully_connected",
    "lstm",
    "mul",
    "pad",
    "pooling",
    "prelu",
    "quantize_and_dequantize",
    "relu",
    "mean",
    "resampler",
    "reshape",
    "resize",
    "slice",
    "softmax",
    "space_to_depth",
    "tile",
    "transpose_conv",
]

NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "max_unpooling",
]

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    visibility = ["//visibility:public"],
    deps = [":" + op_name for op_name in TFLITE_GPU_BINARY_RELEASE_OPERATORS] +
           select({
               "//tensorflow/lite/delegates/gpu:tflite_gpu_binary_release": [],
               "//conditions:default": NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS,
           }) + [
        ":custom_registry",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_util"]})
