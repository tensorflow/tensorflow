load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "op_resolver_internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":error_reporter.h",
        ":op_resolver.h",
        ":op_resolver_internal.h",
        ":verifier.h",
    ],
    visibility = [
        "//tensorflow/lite:__pkg__",
    ],
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
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":error_reporter",
        ":op_resolver",
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/core/c:common",
        # TODO(b/158301698): consider moving internal:compatibility to a more
        # central location.
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
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
    copts = tflite_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":error_reporter",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "error_reporter",
    hdrs = [
        "error_reporter.h",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
    ],
)

cc_library(
    name = "verifier",
    hdrs = [
        "verifier.h",
        "//tensorflow/compiler/mlir/lite/core/api:verifier.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "//tensorflow/compiler/mlir/lite/core/api:verifier",
    ],
)

cc_library(
    name = "op_resolver_internal",
    hdrs = ["op_resolver_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = op_resolver_internal_visibility_allowlist() + [
        "//tensorflow/lite:__pkg__",
        "//tensorflow/lite/java/src/main/native:__pkg__",
    ],
    deps = [":op_resolver"],
)

cc_test(
    name = "op_resolver_test",
    size = "small",
    srcs = ["op_resolver_test.cc"],
    deps = [
        ":api",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "op_resolver_internal_test",
    size = "small",
    srcs = ["op_resolver_internal_test.cc"],
    deps = [
        ":op_resolver",
        ":op_resolver_internal",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "flatbuffer_conversions_test",
    size = "small",
    srcs = ["flatbuffer_conversions_test.cc"],
    deps = [
        ":api",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)
