load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite/core:special_rules.bzl", "delegate_registry_visibility_allowlist")
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + delegate_registry_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
    ],
)

cc_library(
    name = "stable_delegate_registry",
    hdrs = ["stable_delegate_registry.h"],
    visibility = experimental_acceleration_api_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration:stable_delegate_registry",
    ],
)
