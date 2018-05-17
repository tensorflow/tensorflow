package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/contrib/lite:special_rules.bzl", "tflite_portable_test_suite")

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
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/contrib/lite:builtin_op_data",
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
        "optimized/cblas_conv.h",
        "optimized/cblas_reference.h",
        "optimized/eigen_spatial_convolutions.h",
        "optimized/eigen_tensor_reduced_instantiations_oss.h",
        "optimized/multithreaded_conv.h",
        "tensor.h",
    ],
    deps = [
        ":optimized_base",
        ":types",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:context",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":reference",
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
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
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
        "reference/reference_ops.h",
    ],
    deps = [
        ":quantization_util",
        ":round",
        ":strided_slice_logic",
        ":types",
        "//third_party/eigen3",
        "@gemmlowp",
        "//tensorflow/contrib/lite:builtin_op_data",
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
    name = "reference",
    hdrs = ["tensor.h"],
    deps = [
        ":types",
        "//tensorflow/contrib/lite:context",
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
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite/kernels:activation_functor",
        "//tensorflow/contrib/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "neon_tensor_utils",
    srcs = [
        "optimized/neon_tensor_utils.cc",
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
        ":portable_tensor_utils",
        ":round",
        ":types",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite/kernels:activation_functor",
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
        "//tensorflow/contrib/lite:builtin_op_data",
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
        "//tensorflow/contrib/lite/kernels:activation_functor",
        "//tensorflow/contrib/lite:builtin_op_data",
        "@arm_neon_2_x86_sse",
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
        "//conditions:default": [
            ":portable_tensor_utils",
        ],
    }),
)

cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    copts = NEON_FLAGS_IF_APPLICABLE,
    linkopts = select({
        "//tensorflow:android": [
            "-fPIE -pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":tensor_utils",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite/kernels:test_util",
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
