# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

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
    name = "person_detect_model_data",
    srcs = [
        "person_detect_model_data.cc",
    ],
    hdrs = [
        "person_detect_model_data.h",
    ],
)

cc_library(
    name = "simple_images_test_data",
    srcs = [
        "no_person_image_data.cc",
        "person_image_data.cc",
    ],
    hdrs = [
        "no_person_image_data.h",
        "person_image_data.h",
    ],
    deps = [
        ":model_settings",
    ],
)

cc_library(
    name = "image_provider",
    srcs = [
        "image_provider.cc",
    ],
    hdrs = [
        "image_provider.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "image_provider_test",
    srcs = [
        "image_provider_test.cc",
    ],
    deps = [
        ":image_provider",
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "detection_responder",
    srcs = [
        "detection_responder.cc",
    ],
    hdrs = [
        "detection_responder.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "detection_responder_test",
    srcs = [
        "detection_responder_test.cc",
    ],
    deps = [
        ":detection_responder",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "person_detection",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    deps = [
        ":detection_responder",
        ":image_provider",
        ":model_settings",
        ":person_detect_model_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
