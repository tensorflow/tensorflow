load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.jxl"]) + glob(["*.png"]),
    visibility = ["//tensorflow/core/lib/jxl:__pkg__"],
)
