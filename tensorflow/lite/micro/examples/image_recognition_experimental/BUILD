# Description:
#   TensorFlow Lite for Microcontrollers image recognition example.

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "image_model_data",
    srcs = [
        "first_10_cifar_images.cc",
        "image_recognition_model.cc",
    ],
    hdrs = [
        "first_10_cifar_images.h",
        "image_recognition_model.h",
        "util.h",
    ],
)

tflite_micro_cc_test(
    name = "image_recognition_test",
    srcs = ["image_recognition_test.cc"],
    deps = [
        ":image_model_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
