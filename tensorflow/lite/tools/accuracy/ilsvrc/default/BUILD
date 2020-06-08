package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "custom_delegates",
    hdrs = ["custom_delegates.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools/evaluation/stages:image_classification_stage",
    ],
)
