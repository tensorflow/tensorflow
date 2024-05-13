load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "nnapi_plugin_impl_visibility_allowlist", "xnnpack_plugin_impl_visibility_allowlist")
load("//tensorflow/lite/core:special_rules.bzl", "delegate_registry_visibility_allowlist")
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")

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
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "nnapi_plugin",
    srcs = ["nnapi_plugin.cc"],
    hdrs = ["nnapi_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = nnapi_plugin_impl_visibility_allowlist() + [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":delegate_registry",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/nnapi:nnapi_implementation_headers",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "nnapi_plugin_test",
    srcs = ["nnapi_plugin_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":delegate_registry",
        ":nnapi_plugin",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate_mock_test",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
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
        "//tensorflow/lite/core/acceleration/configuration/c:stable_delegate",
        "//tensorflow/lite/core/shims:tflite_use_opaque_delegate",  # buildcleaner: keep
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "stable_delegate_registry_test",
    srcs = ["stable_delegate_registry_test.cc"],
    deps = [
        ":stable_delegate_registry",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xnnpack_plugin",
    srcs = ["xnnpack_plugin.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = xnnpack_plugin_impl_visibility_allowlist() + [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "xnnpack_plugin_test",
    srcs = ["xnnpack_plugin_test.cc"],
    deps = [
        ":xnnpack_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)
