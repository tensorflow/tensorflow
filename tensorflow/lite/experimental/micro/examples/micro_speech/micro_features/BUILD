# Library for generating feature vectors from audio data

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
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
    name = "static_alloc",
    hdrs = ["static_alloc.h"],
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
        ":micro_model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
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
