# Description:
#   TensorFlow Lite for Microcontrollers "gesture recognition" example.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "gesture_recognition_model_data",
    srcs = [
        "gesture_recognition_model_data.cc",
    ],
    hdrs = [
        "gesture_recognition_model_data.h",
    ],
)

cc_library(
    name = "sample_feature_data",
    srcs = [
        "angle_micro_features_data.cc",
        "circle_micro_features_data.cc",
    ],
    hdrs = [
        "angle_micro_features_data.h",
        "circle_micro_features_data.h",
    ],
)

tflite_micro_cc_test(
    name = "gesture_recognition_test",
    srcs = [
        "gesture_recognition_test.cc",
    ],
    deps = [
        ":gesture_recognition_model_data",
        ":sample_feature_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
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
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "accelerometer_handler_test",
    srcs = [
        "accelerometer_handler_test.cc",
    ],
    deps = [
        ":accelerometer_handler",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
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
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "output_handler_test",
    srcs = [
        "output_handler_test.cc",
    ],
    deps = [
        ":output_handler",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "gesture_recognition",
    srcs = [
        "main.cc",
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/gesture_recognition:accelerometer_handler",
        "//tensorflow/lite/experimental/micro/examples/gesture_recognition:gesture_recognition_model_data",
        "//tensorflow/lite/experimental/micro/examples/gesture_recognition:output_handler",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
