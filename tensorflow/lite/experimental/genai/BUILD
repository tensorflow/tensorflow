# Custom Ops useful for GenAI models.
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "kvcache",
    srcs = [
        "kvcache.cc",
    ],
    hdrs = ["kvcache.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "kvcache_test",
    srcs = ["kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":kvcache",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
