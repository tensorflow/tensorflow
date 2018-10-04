# Description:
#   TensorFlow Lite microcontroller example.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/contrib/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

tflite_micro_cc_test(
    name = "micro_speech_test",
    srcs = [
        "micro_speech_test.cc",
        "tiny_conv_model_data.cc",
        "tiny_conv_model_data.h",
    ],
    deps = [
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/experimental/micro:micro_framework",
        "//tensorflow/contrib/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/contrib/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/contrib/lite/experimental/micro/testing:micro_test",
        "//tensorflow/contrib/lite/schema:schema_fbs",
    ],
)
