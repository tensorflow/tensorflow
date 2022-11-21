# Data service test data.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.pbtxt"]),
    visibility = ["//tensorflow/core/data/service:__pkg__"],
)
