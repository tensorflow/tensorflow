# Library for generating feature vectors from audio data

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
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
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
        "//tensorflow/lite/micro:micro_error_reporter",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/examples/micro_speech:audio_sample_test_data",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
