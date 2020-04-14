# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "cc_library",
    "micro_copts",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "sine_model_data",
    srcs = [
        "sine_model_data.cc",
    ],
    hdrs = [
        "sine_model_data.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
)

tflite_micro_cc_test(
    name = "hello_world_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/examples/hello_world:sine_model_data",
        "//tensorflow/lite/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
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
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
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
    copts = micro_copts(),
)

cc_binary(
    name = "hello_world",
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
        ":constants",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/examples/hello_world:sine_model_data",
        "//tensorflow/lite/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
