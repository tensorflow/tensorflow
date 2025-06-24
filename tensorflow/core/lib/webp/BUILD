load("//tensorflow:tensorflow.bzl", "if_google")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "webp_io",
    srcs = ["webp_io.cc"],
    hdrs = ["webp_io.h"],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@libwebp//:webp",
    ] + if_google([
        "@libwebp//:webp_demux",
    ]),
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/webp/testdata:webp_testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)
