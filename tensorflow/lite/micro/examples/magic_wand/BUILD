# Description:
#   TensorFlow Lite for Microcontrollers "gesture recognition" example.

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "magic_wand_model_data",
    srcs = [
        "magic_wand_model_data.cc",
    ],
    hdrs = [
        "magic_wand_model_data.h",
    ],
)

cc_library(
    name = "sample_feature_data",
    srcs = [
        "ring_micro_features_data.cc",
        "slope_micro_features_data.cc",
    ],
    hdrs = [
        "ring_micro_features_data.h",
        "slope_micro_features_data.h",
    ],
)

tflite_micro_cc_test(
    name = "magic_wand_test",
    srcs = [
        "magic_wand_test.cc",
    ],
    deps = [
        ":magic_wand_model_data",
        ":sample_feature_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "constants",
    srcs = [
        "constants.cc",
    ],
    hdrs = [
        "constants.h",
    ],
)

cc_library(
    name = "accelerometer_handler",
    srcs = [
        "accelerometer_handler.cc",
    ],
    hdrs = [
        "accelerometer_handler.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "accelerometer_handler_test",
    srcs = [
        "accelerometer_handler_test.cc",
    ],
    deps = [
        ":accelerometer_handler",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "gesture_predictor",
    srcs = [
        "gesture_predictor.cc",
    ],
    hdrs = [
        "gesture_predictor.h",
    ],
    deps = [
        ":constants",
    ],
)

tflite_micro_cc_test(
    name = "gesture_predictor_test",
    srcs = [
        "gesture_predictor_test.cc",
    ],
    deps = [
        ":constants",
        ":gesture_predictor",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "output_handler",
    srcs = [
        "output_handler.cc",
    ],
    hdrs = [
        "output_handler.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "output_handler_test",
    srcs = [
        "output_handler_test.cc",
    ],
    deps = [
        ":output_handler",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "magic_wand",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        ":accelerometer_handler",
        ":constants",
        ":gesture_predictor",
        ":magic_wand_model_data",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
