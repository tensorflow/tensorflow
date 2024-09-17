# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

cc_library(
    name = "util",
    srcs = [
    ],
    hdrs = [
        "util.h",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/experimental/shlo/legacy:float",
    ],
)

cc_binary(
    name = "shlo_benchmark",
    srcs = [
        "shlo_benchmark.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:shlo",
        "//tensorflow/lite/experimental/shlo/legacy/test:util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "xnn_benchmark",
    srcs = [
        "xnn_benchmark.cc",
    ],
    data = [
    ],
    tags = ["no_oss"],
    deps = [
        ":util",
        "//tensorflow/lite/experimental/shlo/legacy:float",
        "@XNNPACK",
        "@com_google_absl//absl/log",
        "@com_google_benchmark//:benchmark",
    ],
)
