# Description:
# WebP test data.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "webp_testdata",
    srcs = glob(["*.webp"]),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/webp:__pkg__",
    ],
)
