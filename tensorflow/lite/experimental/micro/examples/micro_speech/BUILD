# Description:
#   TensorFlow Lite microcontroller example.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "model_settings",
    srcs = [
        "model_settings.cc",
    ],
    hdrs = [
        "model_settings.h",
    ],
)

cc_library(
    name = "tiny_conv_model_data",
    srcs = [
        "tiny_conv_model_data.cc",
    ],
    hdrs = [
        "tiny_conv_model_data.h",
    ],
)

cc_library(
    name = "features_test_data",
    srcs = [
        "no_features_data.cc",
        "yes_features_data.cc",
    ],
    hdrs = [
        "no_features_data.h",
        "yes_features_data.h",
    ],
)

tflite_micro_cc_test(
    name = "micro_speech_test",
    srcs = [
        "micro_speech_test.cc",
    ],
    deps = [
        ":features_test_data",
        ":tiny_conv_model_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "preprocessor_test_data",
    srcs = [
        "no_30ms_sample_data.cc",
        "no_power_spectrum_data.cc",
        "yes_30ms_sample_data.cc",
        "yes_power_spectrum_data.cc",
    ],
    hdrs = [
        "no_30ms_sample_data.h",
        "no_power_spectrum_data.h",
        "yes_30ms_sample_data.h",
        "yes_power_spectrum_data.h",
    ],
)

cc_library(
    name = "preprocessor_reference",
    srcs = [
        "preprocessor.cc",
    ],
    hdrs = [
        "preprocessor.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "preprocessor_reference_test",
    srcs = [
        "preprocessor_test.cc",
    ],
    deps = [
        ":model_settings",
        ":preprocessor_reference",
        ":preprocessor_test_data",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "preprocessor_fixed",
    srcs = [
        "fixed_point/preprocessor.cc",
    ],
    hdrs = [
        "preprocessor.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "preprocessor_fixed_test",
    srcs = [
        "preprocessor_test.cc",
    ],
    deps = [
        ":model_settings",
        ":preprocessor_fixed",
        ":preprocessor_test_data",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "audio_provider",
    srcs = [
        "audio_provider.cc",
    ],
    hdrs = [
        "audio_provider.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "audio_provider_test",
    srcs = [
        "audio_provider_test.cc",
    ],
    deps = [
        ":audio_provider",
        ":model_settings",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "feature_provider",
    srcs = [
        "feature_provider.cc",
    ],
    hdrs = [
        "feature_provider.h",
    ],
    deps = [
        ":audio_provider",
        ":model_settings",
        ":preprocessor_reference",
        ":timer",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "feature_provider_test",
    srcs = [
        "feature_provider_test.cc",
    ],
    deps = [
        ":audio_provider",
        ":feature_provider",
        ":model_settings",
        ":timer",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "timer",
    srcs = [
        "timer.cc",
    ],
    hdrs = [
        "timer.h",
    ],
)

tflite_micro_cc_test(
    name = "timer_test",
    srcs = [
        "timer_test.cc",
    ],
    deps = [
        ":timer",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "micro_speech",
    srcs = [
        "main.cc",
    ],
    deps = [
        ":audio_provider",
        ":feature_provider",
        ":features_test_data",
        ":model_settings",
        ":preprocessor_reference",
        ":timer",
        ":tiny_conv_model_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
