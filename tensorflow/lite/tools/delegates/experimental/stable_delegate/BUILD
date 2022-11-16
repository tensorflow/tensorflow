# Provides stable ABI delegate.
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "delegate_provider",
    srcs = ["stable_delegate_provider.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/core/shims:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Stable ABI delegate does not support Windows because the shared library loader hasn't been
        # extended to support Windows.
        "//tensorflow:windows": [],
        "//conditions:default": [
            "//tensorflow/lite/core/shims:delegate_plugin",
            "//tensorflow/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
            "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
            "//tensorflow/lite/experimental/acceleration/configuration/c:stable_delegate",
        ],
    }),
    # Statically registers itself with DelegateProviderRegistrar.
    alwayslink = 1,
)

cc_test(
    name = "delegate_provider_test",
    size = "small",
    srcs = ["stable_delegate_provider_test.cc"],
    data = ["//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate"],
    # Disable the test on Windows as the shared library loader doesn't support it.
    tags = ["no-windows"],
    deps = [
        ":delegate_provider",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:delegate_provider_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
