load("//tensorflow/lite/experimental/shlo:build_defs.bzl", "shlo_cc_library", "shlo_cc_test")

package(default_applicable_licenses = ["//tensorflow:license"])

shlo_cc_library(
    name = "util",
    srcs = [
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

shlo_cc_test(
    name = "broadcast_in_dim",
    srcs = [
        "broadcast_in_dim_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "clamp",
    srcs = [
        "clamp_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "compare",
    srcs = [
        "compare_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "concatenate",
    srcs = [
        "concatenate_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "elementwise_binary",
    srcs = [
        "elementwise_binary_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "elementwise_unary",
    srcs = [
        "elementwise_unary_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "is_finite",
    srcs = [
        "is_finite_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "select",
    srcs = [
        "select_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

shlo_cc_test(
    name = "uniform_dequantize_quantize",
    srcs = [
        "uniform_dequantize_quantize_test.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
