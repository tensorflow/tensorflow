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
    name = "png_io",
    srcs = ["png_io.cc"],
    hdrs = ["png_io.h"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:png",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@png",
        "@zlib",
    ],
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/png/testdata:png_testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)
