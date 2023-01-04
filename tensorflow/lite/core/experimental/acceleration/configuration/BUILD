package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    srcs = ["delegate_registry.cc"],
    hdrs = ["delegate_registry.h"],
    visibility = [
        "//tensorflow/lite/experimental/acceleration/configuration:__subpackages__",
    ],
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
    deps = [
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:stable_delegate",
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
