# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "sine_model_data",
    srcs = [
        "sine_model_data.cc",
    ],
    hdrs = [
        "sine_model_data.h",
    ],
)

tflite_micro_cc_test(
    name = "hello_world_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/hello_world:sine_model_data",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
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
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
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
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/hello_world:sine_model_data",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
