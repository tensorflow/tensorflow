# Description:
#   TensorFlow Lite microcontroller example.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

tflite_micro_cc_test(
    name = "micro_speech_test",
    srcs = [
        "micro_speech_test.cc",
        "no_features_data.cc",
        "no_features_data.h",
        "tiny_conv_model_data.cc",
        "tiny_conv_model_data.h",
        "yes_features_data.cc",
        "yes_features_data.h",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflite_micro_cc_test(
    name = "preprocessor_reference_test",
    srcs = [
        "no_30ms_sample_data.cc",
        "no_30ms_sample_data.h",
        "no_power_spectrum_data.cc",
        "no_power_spectrum_data.h",
        "preprocessor.cc",
        "preprocessor.h",
        "preprocessor_test.cc",
        "yes_30ms_sample_data.cc",
        "yes_30ms_sample_data.h",
        "yes_power_spectrum_data.cc",
        "yes_power_spectrum_data.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "preprocessor_fixed_test",
    srcs = [
        "fixed_point/preprocessor.cc",
        "no_30ms_sample_data.cc",
        "no_30ms_sample_data.h",
        "no_power_spectrum_data.cc",
        "no_power_spectrum_data.h",
        "preprocessor.h",
        "preprocessor_test.cc",
        "yes_30ms_sample_data.cc",
        "yes_30ms_sample_data.h",
        "yes_power_spectrum_data.cc",
        "yes_power_spectrum_data.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)
