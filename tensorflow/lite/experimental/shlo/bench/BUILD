load("//tensorflow/lite/experimental/shlo:build_defs.bzl", "shlo_cc_binary", "shlo_cc_library")

package(default_applicable_licenses = ["//tensorflow:license"])

shlo_cc_library(
    name = "util",
    srcs = [
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/shlo:float",
    ],
)

shlo_cc_binary(
    name = "shlo_benchmark",
    srcs = [
        "shlo_benchmark.cc",
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo",
        "//tensorflow/lite/experimental/shlo:debug",
        "//tensorflow/lite/experimental/shlo/test:util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

shlo_cc_binary(
    name = "xnn_benchmark",
    srcs = [
        "xnn_benchmark.cc",
    ],
    data = [
    ],
    deps = [
        ":util",
        "//base",
        "//tensorflow/lite/experimental/shlo:float",
        "@XNNPACK",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)
