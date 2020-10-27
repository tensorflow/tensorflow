load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "micro_copts")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "api",
    srcs = [
        "flatbuffer_conversions.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "op_resolver.h",
        "profiler.h",
        "tensor_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + micro_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":error_reporter",
        ":op_resolver",
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/c:common",
        # TODO(b/158301698): consider moving internal:compatibility to a more
        # central location.
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
    ],
)

# We define separate targets for "op_resolver" and "error_reporter",
# even though those headers are also exported by the "api" target,
# so that targets which only want to depend on these small abstract base
# class modules can express more fine-grained dependencies without
# pulling in tensor_utils and flatbuffer_conversions.

cc_library(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + micro_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":error_reporter",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + micro_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [],
)

cc_library(
    name = "verifier",
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + micro_copts(),
    visibility = ["//visibility:public"],
    deps = [":error_reporter"],
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
        "//tensorflow/lite/schema:schema_conversion_utils",
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
