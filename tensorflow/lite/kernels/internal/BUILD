load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

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
    ":armv7a": [
        "-O3",
        "-mfpu=neon",
    ],
    "//conditions:default": [
        "-O3",
    ],
})

cc_library(
    name = "types",
    srcs = [],
    hdrs = [
        "compatibility.h",
        "types.h",
    ],
    deps = [
        "//tensorflow/lite/kernels:op_macros",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "legacy_types",
    srcs = [],
    hdrs = [
        "compatibility.h",
        "legacy_types.h",
        "types.h",
    ],
    deps = [
        "//tensorflow/lite/kernels:op_macros",
        "@com_google_absl//absl/base:core_headers",
    ],
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

cc_library(
    name = "optimized_base",
    srcs = [],
    hdrs = [
        "common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/optimized_ops.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        ":strided_slice_logic",
        ":types",
        ":reference_base",
        ":round",
        ":tensor_utils",
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/lite/c:c_api_internal",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "legacy_optimized_base",
    srcs = [],
    hdrs = [
        "common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/legacy_optimized_ops.h",
        "optimized/optimized_ops.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor_utils",
        ":types",
        ":legacy_types",
        ":legacy_reference_base",
        ":round",
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/lite/c:c_api_internal",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "optimized",
    hdrs = [
        "optimized/eigen_spatial_convolutions.h",
        "optimized/eigen_tensor_reduced_instantiations_oss.h",
        "optimized/multithreaded_conv.h",
        # FIXME(petewarden) - This should be removed, since it's a header from the
        # :tensor dependency below.
        "tensor.h",
    ],
    deps = [
        ":optimized_base",
        ":tensor",
        ":types",
        "//tensorflow/lite/c:c_api_internal",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":tensor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "round",
    srcs = [],
    hdrs = ["round.h"],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = [
        "compatibility.h",
        "quantization_util.h",
    ],
    deps = [
        ":round",
        ":types",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":quantization_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    deps = [
        ":types",
    ],
)

cc_library(
    name = "reference_base",
    srcs = [],
    hdrs = [
        "common.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/fully_connected.h",
        "reference/integer_ops/dequantize.h",
        "reference/integer_ops/pooling.h",
        "reference/integer_ops/softmax.h",
        "reference/reference_ops.h",
        "reference/softmax.h",
    ],
    deps = [
        ":quantization_util",
        ":round",
        ":strided_slice_logic",
        ":types",
        "@gemmlowp",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:op_macros",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "legacy_reference_base",
    srcs = [],
    hdrs = [
        "common.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/fully_connected.h",
        "reference/legacy_reference_ops.h",
        "reference/reference_ops.h",
        "reference/softmax.h",
    ],
    deps = [
        ":quantization_util",
        ":round",
        ":strided_slice_logic",
        ":legacy_types",
        ":types",
        "@gemmlowp",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:op_macros",
    ] + select({
        ":haswell": tflite_deps_intel,
        ":ios_x86_64": tflite_deps_intel,
        ":k8": tflite_deps_intel,
        ":x86": tflite_deps_intel,
        ":x86_64": tflite_deps_intel,
        ":darwin": tflite_deps_intel,
        ":darwin_x86_64": tflite_deps_intel,
        ":freebsd": tflite_deps_intel,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensor",
    hdrs = [
        "tensor.h",
        "tensor_ctypes.h",
    ],
    deps = [
        ":types",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

# Deprecated version of :tensor, kept for backwards compatibility.
cc_library(
    name = "reference",
    hdrs = [
        "tensor.h",
        "tensor_ctypes.h",
    ],
    deps = [
        ":types",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = [
        "reference/portable_tensor_utils.cc",
    ],
    hdrs = [
        "reference/portable_tensor_utils.h",
    ],
    deps = [
        ":round",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:activation_functor",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "neon_tensor_utils",
    srcs = [
        "optimized/neon_tensor_utils.cc",
        "reference/portable_tensor_utils.cc",
        "reference/portable_tensor_utils.h",
    ],
    hdrs = [
        "common.h",
        "compatibility.h",
        "optimized/cpu_check.h",
        "optimized/neon_tensor_utils.h",
        "optimized/tensor_utils_impl.h",
    ],
    copts = NEON_FLAGS_IF_APPLICABLE + HARD_FP_FLAGS_IF_APPLICABLE,
    deps = [
        ":cpu_check",
        ":round",
        ":types",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:activation_functor",
        "//tensorflow/lite/kernels:op_macros",
        "@arm_neon_2_x86_sse",
        "@gemmlowp",
    ],
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    deps = [
        ":tensor_utils",
        "//tensorflow/lite/c:c_api_internal",
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
        "common.h",
        "compatibility.h",
        "optimized/cpu_check.h",
        "optimized/neon_tensor_utils.h",
        "optimized/tensor_utils_impl.h",
        "reference/portable_tensor_utils.h",
        "tensor_utils.h",
        "types.h",
    ],
    copts = NEON_FLAGS_IF_APPLICABLE,
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/lite/c:c_api_internal",
        "@arm_neon_2_x86_sse",
        "//tensorflow/lite/kernels:op_macros",
        "@gemmlowp",
    ] + select({
        ":arm": [
            ":neon_tensor_utils",
        ],
        ":arm64-v8a": [
            ":neon_tensor_utils",
        ],
        ":armeabi-v7a": [
            ":neon_tensor_utils",
        ],
        ":armv7a": [
            ":neon_tensor_utils",
        ],
        ":haswell": [
            ":neon_tensor_utils",
        ],
        ":ios_armv7": [
            ":neon_tensor_utils",
        ],
        ":ios_arm64": [
            ":neon_tensor_utils",
        ],
        ":ios_x86_64": [
            ":neon_tensor_utils",
        ],
        ":x86_64": [
            ":neon_tensor_utils",
        ],
        ":x86": [
            ":neon_tensor_utils",
        ],
        ":k8": [
            ":neon_tensor_utils",
        ],
        ":darwin": [
            ":neon_tensor_utils",
        ],
        ":darwin_x86_64": [
            ":neon_tensor_utils",
        ],
        "//conditions:default": [
            ":portable_tensor_utils",
        ],
    }),
)

cc_library(
    name = "test_util",
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":types",
        "//tensorflow/lite:string",
    ],
)

# TODO(b/122597976): Eliminate TF dependency from lite/kernels:test_util,
# in turn eliminating the need to use tf_cc_test for any dependent tests.
tf_cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-fPIE -pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":tensor_utils",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_float_test",
    srcs = ["depthwiseconv_float_test.cc"],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_quantized_test",
    srcs = ["depthwiseconv_quantized_test.cc"],
    shard_count = 2,
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = ["resize_bilinear_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
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
    tags = [
        "tflite_not_portable",
    ],
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
    shard_count = 3,
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
    name = "logsoftmax_quantized_test",
    timeout = "long",
    srcs = [
        "logsoftmax_quantized_test.cc",
    ],
    shard_count = 3,
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
    name = "log_quantized_test",
    srcs = ["log_quantized_test.cc"],
    deps = [
        ":optimized_base",
        ":reference_base",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpu_check",
    hdrs = [
        "optimized/cpu_check.h",
    ],
    deps = [
    ] + select(
        {
            "//tensorflow:android": [
                "@androidndk//:cpufeatures",
            ],
            "//conditions:default": [],
        },
    ),
)

cc_test(
    name = "batch_to_space_nd_test",
    srcs = ["batch_to_space_nd_test.cc"],
    deps = [
        ":optimized_base",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(["optimized/eigen_tensor_reduced_instantiations_oss.h"])

tflite_portable_test_suite()
