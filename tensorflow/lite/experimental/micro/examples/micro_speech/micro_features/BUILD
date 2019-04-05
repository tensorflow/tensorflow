# Library for generating feature vectors from audio data

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "micro_model_settings",
    srcs = [
        "micro_model_settings.cc",
    ],
    hdrs = [
        "micro_model_settings.h",
    ],
)

cc_library(
    name = "tiny_conv_micro_features_model_data",
    srcs = [
        "tiny_conv_micro_features_model_data.cc",
    ],
    hdrs = [
        "tiny_conv_micro_features_model_data.h",
    ],
)

cc_library(
    name = "micro_features_test_data",
    srcs = [
        "no_micro_features_data.cc",
        "yes_micro_features_data.cc",
    ],
    hdrs = [
        "no_micro_features_data.h",
        "yes_micro_features_data.h",
    ],
)

cc_library(
    name = "bits",
    hdrs = ["bits.h"],
)

cc_library(
    name = "static_alloc",
    hdrs = ["static_alloc.h"],
)

cc_library(
    name = "fft",
    srcs = [
        "fft.cc",
        "fft_util.cc",
    ],
    hdrs = [
        "fft.h",
        "fft_util.h",
    ],
    deps = [
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "@kissfft//:kiss_fftr_16",
    ],
)

cc_library(
    name = "filterbank",
    srcs = [
        "filterbank.cc",
        "filterbank_util.cc",
    ],
    hdrs = [
        "filterbank.h",
        "filterbank_util.h",
    ],
    deps = [
        ":bits",
        ":fft",
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "frontend",
    srcs = [
        "frontend.cc",
        "frontend_util.cc",
    ],
    hdrs = [
        "frontend.h",
        "frontend_util.h",
    ],
    deps = [
        ":bits",
        ":fft",
        ":filterbank",
        ":log_scale",
        ":micro_model_settings",
        ":noise_reduction",
        ":pcan_gain_control",
        ":window",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "log_scale",
    srcs = [
        "log_lut.cc",
        "log_scale.cc",
        "log_scale_util.cc",
    ],
    hdrs = [
        "log_lut.h",
        "log_scale.h",
        "log_scale_util.h",
    ],
    deps = [
        ":bits",
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "noise_reduction",
    srcs = [
        "noise_reduction.cc",
        "noise_reduction_util.cc",
    ],
    hdrs = [
        "noise_reduction.h",
        "noise_reduction_util.h",
    ],
    deps = [
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "pcan_gain_control",
    srcs = [
        "pcan_gain_control.cc",
        "pcan_gain_control_util.cc",
    ],
    hdrs = [
        "pcan_gain_control.h",
        "pcan_gain_control_util.h",
    ],
    deps = [
        ":bits",
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "window",
    srcs = [
        "window.cc",
        "window_util.cc",
    ],
    hdrs = [
        "window.h",
        "window_util.h",
    ],
    deps = [
        ":micro_model_settings",
        ":static_alloc",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "micro_features_generator",
    srcs = [
        "micro_features_generator.cc",
    ],
    hdrs = [
        "micro_features_generator.h",
    ],
    deps = [
        ":frontend",
        ":micro_model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "micro_features_generator_test_data",
    srcs = [
        "no_feature_data_slice.cc",
        "yes_feature_data_slice.cc",
    ],
    hdrs = [
        "no_feature_data_slice.h",
        "yes_feature_data_slice.h",
    ],
)

tflite_micro_cc_test(
    name = "fft_test",
    srcs = ["fft_test.cc"],
    deps = [
        ":fft",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "filterbank_test",
    srcs = ["filterbank_test.cc"],
    deps = [
        ":filterbank",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "frontend_test",
    srcs = ["frontend_test.cc"],
    deps = [
        ":frontend",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "log_scale_test",
    srcs = ["log_scale_test.cc"],
    deps = [
        ":log_scale",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "noise_reduction_test",
    srcs = ["noise_reduction_test.cc"],
    deps = [
        ":noise_reduction",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "pcan_gain_control_test",
    srcs = ["pcan_gain_control_test.cc"],
    deps = [
        ":pcan_gain_control",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "window_test",
    srcs = ["window_test.cc"],
    deps = [
        ":window",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_features_generator_test",
    srcs = [
        "micro_features_generator_test.cc",
    ],
    deps = [
        ":micro_features_generator",
        ":micro_features_generator_test_data",
        ":micro_model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech:audio_sample_test_data",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)
