load("//tensorflow:tensorflow.bzl", "transitive_hdrs")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "cc_library", "micro_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tflite_deps_intel = [
    "@arm_neon_2_x86_sse",
]

HARD_FP_FLAGS_IF_APPLICABLE = select({
    "//tensorflow:android_arm": ["-mfloat-abi=softfp"],
    "//tensorflow:android_arm64": ["-mfloat-abi=softfp"],
    "//tensorflow:android_armeabi": ["-mfloat-abi=softfp"],
    "//conditions:default": [],
})

NEON_FLAGS_IF_APPLICABLE = select({
    ":arm": [
        "-O3",
        "-mfpu=neon",
    ],
    ":armeabi-v7a": [
        "-O3",
        "-mfpu=neon",
    ],
    ":armhf": [
        "-O3",
        "-mfpu=neon",
    ],
    ":armv7a": [
        "-O3",
        "-mfpu=neon",
    ],
    "//conditions:default": [
        "-O3",
    ],
})

cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":compatibility",
    ],
)

cc_library(
    name = "legacy_types",
    hdrs = ["legacy_types.h"],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":types",
    ],
)

config_setting(
    name = "aarch64",
    values = {
        "cpu": "aarch64",
    },
)

config_setting(
    name = "arm",
    values = {
        "cpu": "arm",
    },
)

config_setting(
    name = "arm64-v8a",
    values = {
        "cpu": "arm64-v8a",
    },
)

config_setting(
    name = "armhf",
    values = {
        "cpu": "armhf",
    },
)

config_setting(
    name = "armv7a",
    values = {
        "cpu": "armv7a",
    },
)

config_setting(
    name = "armeabi-v7a",
    values = {
        "cpu": "armeabi-v7a",
    },
)

config_setting(
    name = "haswell",
    values = {
        "cpu": "haswell",
    },
)

config_setting(
    name = "ios_x86_64",
    values = {
        "cpu": "ios_x86_64",
    },
)

config_setting(
    name = "ios_armv7",
    values = {
        "cpu": "ios_armv7",
    },
)

config_setting(
    name = "ios_arm64",
    values = {
        "cpu": "ios_arm64",
    },
)

config_setting(
    name = "ios_arm64e",
    values = {
        "cpu": "ios_arm64e",
    },
)

config_setting(
    name = "k8",
    values = {
        "cpu": "k8",
    },
)

config_setting(
    name = "x86",
    values = {
        "cpu": "x86",
    },
)

config_setting(
    name = "x86_64",
    values = {
        "cpu": "x86_64",
    },
)

config_setting(
    name = "darwin",
    values = {
        "cpu": "darwin",
    },
)

config_setting(
    name = "darwin_x86_64",
    values = {
        "cpu": "darwin_x86_64",
    },
)

config_setting(
    name = "freebsd",
    values = {
        "cpu": "freebsd",
    },
)

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

config_setting(
    name = "raspberry_pi_with_neon",
    define_values = {
        "raspberry_pi_with_neon": "true",
    },
    values = {
        "cpu": "armeabi",
    },
)

cc_library(
    name = "common",
    srcs = [],
    hdrs = ["common.h"],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":cppmath",
        ":cpu_check",
        ":types",
        "@gemmlowp//:fixedpoint",
    ],
)

cc_library(
    name = "optimized_base",
    srcs = [],
    hdrs = [
        "optimized/batch_matmul.h",
        "optimized/depthwiseconv_3x3_filter_common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_multithread.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/im2col_utils.h",
        "optimized/integer_ops/add.h",
        "optimized/integer_ops/conv.h",
        "optimized/integer_ops/depthwise_conv.h",
        "optimized/integer_ops/depthwise_conv_3x3_filter.h",
        "optimized/integer_ops/depthwise_conv_hybrid.h",
        "optimized/integer_ops/depthwise_conv_hybrid_3x3_filter.h",
        "optimized/integer_ops/fully_connected.h",
        "optimized/integer_ops/mean.h",
        "optimized/integer_ops/mul.h",
        "optimized/integer_ops/pooling.h",
        "optimized/integer_ops/transpose_conv.h",
        "optimized/optimized_ops.h",
        "optimized/sparse_ops/fully_connected.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cpu_check",
        ":quantization_util",
        ":strided_slice_logic",
        ":types",
        ":reference_base",
        ":cppmath",
        ":tensor",
        ":tensor_utils",
        ":transpose_utils",
        "//third_party/eigen3",
        "@gemmlowp//:fixedpoint",
        "@ruy//ruy/profiler:instrumentation",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_threadpool",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        ":windows": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "legacy_optimized_base",
    srcs = [],
    hdrs = [
        "optimized/depthwiseconv_3x3_filter_common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/im2col_utils.h",
        "optimized/legacy_optimized_ops.h",
        "optimized/optimized_ops.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cpu_check",
        ":optimized_base",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":transpose_utils",
        ":types",
        ":legacy_types",
        ":legacy_reference_base",
        ":cppmath",
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_threadpool",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy/profiler:instrumentation",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        ":windows": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "optimized",
    hdrs = [
        "optimized/eigen_spatial_convolutions.h",
        "optimized/eigen_tensor_reduced_instantiations_oss.h",
        "optimized/multithreaded_conv.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":optimized_base",
        ":tensor",
        ":types",
        "//tensorflow/core/kernels:eigen_spatial_convolutions-inl",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
        "max.h",
        "min.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    build_for_embedded = True,
    copts = tflite_copts() + micro_copts(),
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    deps = [
        ":common",
        ":quantization_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "transpose_utils",
    srcs = [
        "transpose_utils.cc",
    ],
    hdrs = [
        "transpose_utils.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":types",
    ],
)

cc_test(
    name = "transpose_utils_test",
    srcs = ["transpose_utils_test.cc"],
    deps = [
        ":transpose_utils",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":compatibility",
        ":types",
    ],
)

cc_library(
    name = "reference_base",
    srcs = [],
    hdrs = [
        "reference/add.h",
        "reference/arg_min_max.h",
        "reference/batch_matmul.h",
        "reference/binary_function.h",
        "reference/ceil.h",
        "reference/comparisons.h",
        "reference/concatenation.h",
        "reference/conv.h",
        "reference/densify.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/dequantize.h",
        "reference/floor.h",
        "reference/fully_connected.h",
        "reference/integer_ops/add.h",
        "reference/integer_ops/conv.h",
        "reference/integer_ops/depthwise_conv.h",
        "reference/integer_ops/dequantize.h",
        "reference/integer_ops/fully_connected.h",
        "reference/integer_ops/l2normalization.h",
        "reference/integer_ops/log_softmax.h",
        "reference/integer_ops/logistic.h",
        "reference/integer_ops/mean.h",
        "reference/integer_ops/mul.h",
        "reference/integer_ops/pooling.h",
        "reference/integer_ops/tanh.h",
        "reference/integer_ops/transpose_conv.h",
        "reference/l2normalization.h",
        "reference/logistic.h",
        "reference/maximum_minimum.h",
        "reference/mul.h",
        "reference/neg.h",
        "reference/non_max_suppression.h",
        "reference/pad.h",
        "reference/pooling.h",
        "reference/prelu.h",
        "reference/process_broadcast_shapes.h",
        "reference/quantize.h",
        "reference/reduce.h",
        "reference/reference_ops.h",
        "reference/requantize.h",
        "reference/resize_nearest_neighbor.h",
        "reference/round.h",
        "reference/softmax.h",
        "reference/sparse_ops/fully_connected.h",
        "reference/strided_slice.h",
        "reference/sub.h",
        "reference/svdf.h",
        "reference/tanh.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
    select_deps = {
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        ":windows": tflite_deps_intel,
        "//conditions:default": [],
    },
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":portable_tensor_utils",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":types",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/tools/optimize/sparsity:format_converter",
        "//third_party/eigen3",
        "@gemmlowp//:fixedpoint",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "legacy_reference_base",
    srcs = [],
    hdrs = [
        "reference/add.h",
        "reference/arg_min_max.h",
        "reference/binary_function.h",
        "reference/ceil.h",
        "reference/comparisons.h",
        "reference/concatenation.h",
        "reference/conv.h",
        "reference/densify.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/dequantize.h",
        "reference/floor.h",
        "reference/fully_connected.h",
        "reference/l2normalization.h",
        "reference/legacy_reference_ops.h",
        "reference/logistic.h",
        "reference/maximum_minimum.h",
        "reference/mul.h",
        "reference/neg.h",
        "reference/pad.h",
        "reference/pooling.h",
        "reference/prelu.h",
        "reference/process_broadcast_shapes.h",
        "reference/quantize.h",
        "reference/reduce.h",
        "reference/reference_ops.h",
        "reference/requantize.h",
        "reference/resize_nearest_neighbor.h",
        "reference/round.h",
        "reference/softmax.h",
        "reference/strided_slice.h",
        "reference/sub.h",
        "reference/tanh.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":quantization_util",
        ":cppmath",
        ":strided_slice_logic",
        ":legacy_types",
        ":tensor",
        ":types",
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:op_macros",
        "@ruy//ruy/profiler:instrumentation",
        "//tensorflow/lite/tools/optimize/sparsity:format_converter",
        "//tensorflow/lite:string_util",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        ":windows": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensor",
    hdrs = [
        "tensor.h",
        "tensor_ctypes.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
    ],
)

# Deprecated version of :tensor, kept for backwards compatibility.
cc_library(
    name = "reference",
    hdrs = [
        "tensor.h",
        "tensor_ctypes.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = [
        "reference/portable_tensor_utils.cc",
    ],
    hdrs = [
        "reference/portable_tensor_utils.h",
        "reference/portable_tensor_utils_impl.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "@gemmlowp",
    ],
)

cc_library(
    name = "neon_tensor_utils",
    srcs = [
        "optimized/neon_tensor_utils.cc",
    ],
    hdrs = [
        "optimized/neon_tensor_utils.h",
        "optimized/neon_tensor_utils_impl.h",
    ],
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE + HARD_FP_FLAGS_IF_APPLICABLE,
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":cpu_check",
        ":portable_tensor_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy",
    ],
)

cc_library(
    name = "sse_tensor_utils",
    srcs = [
        "compatibility.h",
        "optimized/sse_tensor_utils.cc",
    ],
    hdrs = [
        "optimized/sse_tensor_utils.h",
        "optimized/sse_tensor_utils_impl.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":cpu_check",
        ":neon_tensor_utils",
        ":portable_tensor_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    build_for_embedded = True,
    copts = tflite_copts() + micro_copts(),
    deps = [
        ":tensor_utils",
        "//tensorflow/lite/c:common",
    ],
)

# Audio support classes imported directly from TensorFlow.
cc_library(
    name = "audio_utils",
    srcs = [
        "mfcc.cc",
        "mfcc_dct.cc",
        "mfcc_mel_filterbank.cc",
        "spectrogram.cc",
    ],
    hdrs = [
        "mfcc.h",
        "mfcc_dct.h",
        "mfcc_mel_filterbank.h",
        "spectrogram.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
    ],
)

cc_library(
    name = "tensor_utils",
    srcs = [
        "tensor_utils.cc",
    ],
    hdrs = [
        "tensor_utils.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE,
    select_deps = {
        ":aarch64": [
            ":neon_tensor_utils",
        ],
        ":arm": [
            ":neon_tensor_utils",
        ],
        ":arm64-v8a": [
            ":neon_tensor_utils",
        ],
        ":armeabi-v7a": [
            ":neon_tensor_utils",
        ],
        ":armhf": [
            ":neon_tensor_utils",
        ],
        ":armv7a": [
            ":neon_tensor_utils",
        ],
        ":haswell": [
            ":sse_tensor_utils",
        ],
        ":ios_armv7": [
            ":neon_tensor_utils",
        ],
        ":ios_arm64": [
            ":neon_tensor_utils",
        ],
        ":ios_arm64e": [
            ":neon_tensor_utils",
        ],
        ":raspberry_pi_with_neon": [
            ":neon_tensor_utils",
        ],
        ":ios_x86_64": [
            ":sse_tensor_utils",
        ],
        ":x86_64": [
            ":sse_tensor_utils",
        ],
        ":x86": [
            ":sse_tensor_utils",
        ],
        ":k8": [
            ":sse_tensor_utils",
        ],
        ":darwin": [
            ":sse_tensor_utils",
        ],
        ":darwin_x86_64": [
            ":sse_tensor_utils",
        ],
        ":freebsd": [
            ":sse_tensor_utils",
        ],
        ":windows": [":sse_tensor_utils"],
        "//conditions:default": [
            ":portable_tensor_utils",
        ],
    },
    deps = [
        ":cpu_check",
        "//tensorflow/lite/c:common",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "test_util",
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":types",
    ],
)

cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-fPIE -pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        ":common",
        ":quantization_util",
        ":tensor_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_float_test",
    srcs = ["depthwiseconv_float_test.cc"],
    deps = [
        ":common",
        ":cpu_check",
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_quantized_test",
    srcs = [
        "depthwiseconv_quantized_test.cc",
        "optimized/depthwiseconv_uint8_transitional.h",
    ],
    copts = tflite_copts(),
    shard_count = 2,
    deps = [
        ":compatibility",
        ":cpu_check",
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@ruy//ruy:context",
    ],
)

cc_test(
    name = "depthwiseconv_per_channel_quantized_test",
    srcs = [
        "depthwiseconv_per_channel_quantized_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_per_channel_quantized_16x8_test",
    srcs = [
        "depthwiseconv_per_channel_quantized_16x8_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "conv_per_channel_quantized_16x8_test",
    srcs = [
        "conv_per_channel_quantized_16x8_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = ["resize_bilinear_test.cc"],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_nearest_neighbor_test",
    srcs = ["resize_nearest_neighbor_test.cc"],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "softmax_quantized_test",
    timeout = "long",
    srcs = [
        "softmax_quantized_test.cc",
    ],
    shard_count = 4,
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "strided_slice_logic_test",
    timeout = "moderate",
    srcs = [
        "strided_slice_logic_test.cc",
    ],
    shard_count = 4,
    deps = [
        ":strided_slice_logic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logsoftmax_quantized_test",
    timeout = "long",
    srcs = [
        "logsoftmax_quantized_test.cc",
    ],
    shard_count = 4,
    tags = [
        # TODO(b/122242739): Reenable after fixing the flakiness?
        "nomac",
        "tflite_not_portable",
    ],
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "averagepool_quantized_test",
    timeout = "long",
    srcs = [
        "averagepool_quantized_test.cc",
    ],
    shard_count = 1,
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "maxpool_quantized_test",
    timeout = "long",
    srcs = [
        "maxpool_quantized_test.cc",
    ],
    shard_count = 1,
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "log_quantized_test",
    srcs = ["log_quantized_test.cc"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":optimized_base",
        ":reference_base",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpu_check",
    srcs = ["optimized/cpu_check.cc"],
    hdrs = [
        "optimized/cpu_check.h",
        "optimized/neon_check.h",
        "optimized/sse_check.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts(),
    select_deps = {
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        ":windows": tflite_deps_intel,
        "//conditions:default": [],
    },
)

cc_test(
    name = "batch_to_space_nd_test",
    srcs = ["batch_to_space_nd_test.cc"],
    deps = [
        ":optimized_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "non_max_suppression_test",
    srcs = ["non_max_suppression_test.cc"],
    deps = [
        ":reference_base",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "per_channel_dequantize_test",
    srcs = ["per_channel_dequantize_test.cc"],
    deps = [
        ":reference_base",
        ":types",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(["optimized/eigen_tensor_reduced_instantiations_oss.h"])

filegroup(
    name = "optimized_op_headers",
    srcs = glob([
        "optimized/*.h",
    ]),
    visibility = ["//tensorflow/lite:__subpackages__"],
)

filegroup(
    name = "reference_op_headers",
    srcs = glob([
        "reference/*.h",
    ]),
    visibility = ["//tensorflow/lite:__subpackages__"],
)

filegroup(
    name = "headers",
    srcs = glob([
        "*.h",
    ]),
    visibility = ["//tensorflow/lite:__subpackages__"],
)

transitive_hdrs(
    name = "nnapi_external_headers",
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//third_party/eigen3",
        "@gemmlowp",
    ],
)

# ---------------------------------------------------------
# The public target "install_nnapi_extra_headers" is only
# used for external targets that requires exporting optmized
# and reference op headers.

genrule(
    name = "install_nnapi_extra_headers",
    srcs = [
        ":nnapi_external_headers",
        ":headers",
        ":optimized_op_headers",
        ":reference_op_headers",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out*genfiles/}"
      d="$${d#*external/eigen_archive/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi
      fi

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:private"],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": ["//testing/base/public:gunit_main"]})
