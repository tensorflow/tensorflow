# Implementation of StableHLO operations.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "is_finite",
    srcs = ["is_finite.cc"],
    hdrs = ["is_finite.h"],
    deps = [
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:dispatch",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "is_finite_test",
    srcs = ["is_finite_test.cc"],
    deps = [
        ":is_finite",
        "//tensorflow/lite/experimental/shlo:data_type",
        "//tensorflow/lite/experimental/shlo:quantized_tensor_element_type",
        "//tensorflow/lite/experimental/shlo:shape",
        "//tensorflow/lite/experimental/shlo:tensor",
        "//tensorflow/lite/experimental/shlo:tensor_matcher",
        "//tensorflow/lite/experimental/shlo:tensor_with_data",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
