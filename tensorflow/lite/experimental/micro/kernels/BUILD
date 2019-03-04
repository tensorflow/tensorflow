package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "micro_ops",
    srcs = [
        "depthwise_conv.cc",
        "fully_connected.cc",
        "softmax.cc",
    ],
    hdrs = [
    ],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "all_ops_resolver",
    srcs = [
        "all_ops_resolver.cc",
    ],
    hdrs = [
        "all_ops_resolver.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":micro_ops",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        ":all_ops_resolver",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "fully_connected_test",
    srcs = [
        "fully_connected_test.cc",
    ],
    deps = [
        ":all_ops_resolver",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "softmax_test",
    srcs = [
        "softmax_test.cc",
    ],
    deps = [
        ":all_ops_resolver",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)
