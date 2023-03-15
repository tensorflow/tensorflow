load("//tensorflow/core/platform:build_config_root.bzl", "tf_gpu_tests_tags")
load("//tensorflow/lite/core:special_rules.bzl", "delegate_registry_visibility_allowlist")
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    srcs = ["delegate_registry.cc"],
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + delegate_registry_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "stable_delegate_registry",
    srcs = ["stable_delegate_registry.cc"],
    hdrs = ["stable_delegate_registry.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + experimental_acceleration_api_allowlist(),
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:stable_delegate",
        "//tensorflow/lite/core/shims:tflite_use_opaque_delegate",  # buildcleaner: keep
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "stable_delegate_registry_test",
    srcs = ["stable_delegate_registry_test.cc"],
    tags = tf_gpu_tests_tags(),
    deps = [
        ":stable_delegate_registry",
        "@com_google_googletest//:gtest_main",
    ],
)
