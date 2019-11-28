# Description:
#   TensorFlow Lite microcontroller example.

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "simple_model_settings",
    srcs = [
        "simple_features/simple_model_settings.cc",
    ],
    hdrs = [
        "simple_features/simple_model_settings.h",
    ],
)

cc_library(
    name = "tiny_conv_simple_features_model_data",
    srcs = [
        "simple_features/tiny_conv_simple_features_model_data.cc",
    ],
    hdrs = [
        "simple_features/tiny_conv_simple_features_model_data.h",
    ],
)

cc_library(
    name = "simple_features_test_data",
    srcs = [
        "simple_features/no_simple_features_data.cc",
        "simple_features/yes_simple_features_data.cc",
    ],
    hdrs = [
        "simple_features/no_simple_features_data.h",
        "simple_features/yes_simple_features_data.h",
    ],
)

tflite_micro_cc_test(
    name = "micro_speech_test",
    srcs = [
        "micro_speech_test.cc",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_features_test_data",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:tiny_conv_micro_features_model_data",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "audio_sample_test_data",
    srcs = [
        "no_30ms_sample_data.cc",
        "yes_30ms_sample_data.cc",
    ],
    hdrs = [
        "no_30ms_sample_data.h",
        "yes_30ms_sample_data.h",
    ],
)

cc_library(
    name = "audio_large_sample_test_data",
    srcs = [
        "no_1000ms_sample_data.cc",
        "yes_1000ms_sample_data.cc",
    ],
    hdrs = [
        "no_1000ms_sample_data.h",
        "yes_1000ms_sample_data.h",
    ],
)

cc_library(
    name = "simple_features_generator_test_data",
    srcs = [
        "simple_features/no_power_spectrum_data.cc",
        "simple_features/yes_power_spectrum_data.cc",
    ],
    hdrs = [
        "simple_features/no_power_spectrum_data.h",
        "simple_features/yes_power_spectrum_data.h",
    ],
)

cc_library(
    name = "simple_features_generator_reference",
    srcs = [
        "simple_features/simple_features_generator.cc",
    ],
    hdrs = [
        "simple_features/simple_features_generator.h",
    ],
    deps = [
        ":simple_model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "simple_features_generator_reference_test",
    srcs = [
        "simple_features/simple_features_generator_test.cc",
    ],
    deps = [
        ":audio_sample_test_data",
        ":simple_features_generator_reference",
        ":simple_features_generator_test_data",
        ":simple_model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "simple_features_generator_fixed",
    srcs = [
        "simple_features/fixed_point/simple_features_generator.cc",
    ],
    hdrs = [
        "simple_features/simple_features_generator.h",
    ],
    deps = [
        ":simple_model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "simple_features_generator_fixed_test",
    srcs = [
        "simple_features/simple_features_generator_test.cc",
    ],
    deps = [
        ":audio_sample_test_data",
        ":simple_features_generator_fixed",
        ":simple_features_generator_test_data",
        ":simple_model_settings",
        "//tensorflow/lite/c:common",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
    ],
)

cc_library(
    name = "audio_provider_mock",
    srcs = [
        "audio_provider_mock.cc",
    ],
    hdrs = [
        "audio_provider.h",
    ],
    deps = [
        ":audio_large_sample_test_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
    ],
)

tflite_micro_cc_test(
    name = "audio_provider_test",
    srcs = [
        "audio_provider_test.cc",
    ],
    deps = [
        ":audio_provider",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "audio_provider_mock_test",
    srcs = [
        "audio_provider_mock_test.cc",
    ],
    deps = [
        ":audio_large_sample_test_data",
        ":audio_provider_mock",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_features_generator",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
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
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "feature_provider_mock",
    srcs = [
        "feature_provider.cc",
    ],
    hdrs = [
        "feature_provider.h",
    ],
    deps = [
        ":audio_provider_mock",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_features_generator",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
    ],
)

tflite_micro_cc_test(
    name = "feature_provider_mock_test",
    srcs = [
        "feature_provider_mock_test.cc",
    ],
    deps = [
        ":feature_provider_mock",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_features_test_data",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "recognize_commands",
    srcs = [
        "recognize_commands.cc",
    ],
    hdrs = [
        "recognize_commands.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
    ],
)

tflite_micro_cc_test(
    name = "recognize_commands_test",
    srcs = [
        "recognize_commands_test.cc",
    ],
    tags = [
        "no_oss",  # TODO(122853023): Resolve issues and re-enable.
    ],
    deps = [
        ":recognize_commands",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_library(
    name = "command_responder",
    srcs = [
        "command_responder.cc",
    ],
    hdrs = [
        "command_responder.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "command_responder_test",
    srcs = [
        "command_responder_test.cc",
    ],
    deps = [
        ":command_responder",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "micro_speech",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    deps = [
        ":audio_provider",
        ":command_responder",
        ":feature_provider",
        ":recognize_commands",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:tiny_conv_micro_features_model_data",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "micro_speech_mock",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    deps = [
        ":audio_provider_mock",
        ":command_responder",
        ":feature_provider",
        ":recognize_commands",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:micro_model_settings",
        "//tensorflow/lite/experimental/micro/examples/micro_speech/micro_features:tiny_conv_micro_features_model_data",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
