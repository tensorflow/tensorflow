load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "jxl_io",
    srcs = ["jxl_io.cc"],
    hdrs = ["jxl_io.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@com_google_absl//absl/strings",
        "@jpegxl",
        "@jpegxl//:jpegxl_threads",
    ],
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/jxl/testdata:testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)
