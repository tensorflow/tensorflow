load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "cc_library", "micro_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "api",
    srcs = [
        "error_reporter.cc",
        "flatbuffer_conversions.cc",
        "op_resolver.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "op_resolver.h",
        "profiler.h",
        "tensor_utils.h",
    ],
    build_for_embedded = True,
    copts = tflite_copts() + micro_copts(),
    deps = [
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/c:common",
        # TODO(b/158301698): consider moving internal:compatibility to a more
        # central location.
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
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
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest",
    ],
)
