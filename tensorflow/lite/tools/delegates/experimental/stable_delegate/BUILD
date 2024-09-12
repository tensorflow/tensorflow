# Provides stable ABI delegate.
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "delegate_provider",
    srcs = [
        "stable_delegate_provider.cc",
    ],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Stable delegate does not support Windows because the shared library loader hasn't been
        # extended to support Windows.
        "//tensorflow:windows": [],
        "//conditions:default": [
            "//tensorflow/lite/acceleration/configuration:configuration_fbs",
            "//tensorflow/lite/acceleration/configuration/c:delegate_plugin",
            "//tensorflow/lite/acceleration/configuration/c:stable_delegate",
            "//tensorflow/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
            "//tensorflow/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        ],
    }),
    # Statically registers itself with DelegateProviderRegistrar.
    alwayslink = 1,
)

cc_test(
    name = "delegate_provider_test",
    size = "small",
    srcs = ["stable_delegate_provider_test.cc"],
    data = [
        ":test_invalid_settings.json",
        ":test_missing_delegate_path_settings.json",
        ":test_missing_stable_delegate_settings.json",
        ":test_sample_stable_delegate_settings.json",
        ":test_stable_xnnpack_settings.json",
        "//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:tensorflowlite_stable_xnnpack_delegate",
    ],
    # Disable the test on Windows as the shared library loader doesn't support it.
    tags = ["no-windows"],
    deps = [
        ":delegate_provider",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:delegate_provider_lib",
        "@com_google_googletest//:gtest_main",
        "@pthreadpool",
    ],
)

exports_files(
    srcs = [
        "test_invalid_settings.json",
        "test_sample_stable_delegate_settings.json",
    ],
    visibility = ["//visibility:public"],
)
