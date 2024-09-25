# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = [
    ],
    hdrs = [
        "util.h",
    ],
    tags = ["no_oss"],
    visibility = ["//tensorflow/lite/experimental/shlo/legacy/bench:__subpackages__"],
    deps = [
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "broadcast_in_dim",
    srcs = [
        "broadcast_in_dim_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "clamp",
    srcs = [
        "clamp_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "compare",
    srcs = [
        "compare_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "concatenate",
    srcs = [
        "concatenate_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "elementwise_binary",
    srcs = [
        "elementwise_binary_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":matchers",
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "elementwise_unary",
    srcs = [
        "elementwise_unary_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":matchers",
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "iota",
    srcs = [
        "iota_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "is_finite",
    srcs = [
        "is_finite_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "select",
    srcs = [
        "select_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "uniform_dequantize_quantize",
    srcs = [
        "uniform_dequantize_quantize_test.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":matchers",
        "//tensorflow/lite/experimental/shlo/legacy:debug",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "@com_google_googletest//:gtest_main",
    ],
)
