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
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
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
        ":model",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
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
        "//tensorflow/lite/micro:micro_error_reporter",
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
        ":model",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

sh_test(
    name = "hello_world_binary_test",
    srcs = ["hello_world_binary_test.sh"],
    data = [":hello_world"],
)
