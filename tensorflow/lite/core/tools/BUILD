load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_self_contained_libs_test_suite")
load("//tensorflow/lite:special_rules.bzl", "verifier_internal_visibility_allowlist")
load("//tensorflow/lite/core:special_rules.bzl", "verifier_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + verifier_visibility_allowlist(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/tools:verifier_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
        "@flatbuffers//:runtime_cc",
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
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "verifier_internal",
    srcs = ["verifier_internal.cc"],
    hdrs = ["verifier_internal.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = verifier_internal_visibility_allowlist() + [
        "//tensorflow/lite/core/shims:__subpackages__",
        "//tensorflow/lite/java/src/main/native:__pkg__",
        "//tensorflow/lite/tools:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "verifier_internal_test",
    size = "small",
    srcs = ["verifier_internal_test.cc"],
    deps = [
        ":verifier_internal",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
