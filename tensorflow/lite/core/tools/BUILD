load("//tensorflow/lite:special_rules.bzl", "verifier_internal_visibility_allowlist")
load("//tensorflow/lite:build_def.bzl", "tflite_self_contained_libs_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/tools:verifier_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "verifier_test",
    size = "small",
    srcs = ["verifier_test.cc"],
    data = [
        "//tensorflow/lite:testdata/sparse_tensor.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":verifier",
        "//tensorflow/core:framework_lite",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "verifier_internal",
    srcs = ["verifier_internal.cc"],
    hdrs = ["verifier_internal.h"],
    visibility = verifier_internal_visibility_allowlist() + [
        "//tensorflow/lite/tools:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "verifier_internal_test",
    size = "small",
    srcs = ["verifier_internal_test.cc"],
    deps = [
        ":verifier_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
