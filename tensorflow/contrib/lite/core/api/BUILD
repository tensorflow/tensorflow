package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "api",
    srcs = [
        "error_reporter.cc",
        "flatbuffer_conversions.cc",
        "op_resolver.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "op_resolver.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/contrib/lite/c:c_api_internal",
        "//tensorflow/contrib/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "error_reporter_test",
    size = "small",
    srcs = ["error_reporter_test.cc"],
    deps = [
        ":api",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "op_resolver_test",
    size = "small",
    srcs = ["op_resolver_test.cc"],
    deps = [
        ":api",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "flatbuffer_conversions_test",
    size = "small",
    srcs = ["flatbuffer_conversions_test.cc"],
    deps = [
        ":api",
        "//tensorflow/contrib/lite/c:c_api_internal",
        "@com_google_googletest//:gtest",
    ],
)
