# Description:
# JPEG test data packages.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.jpg"]),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ],
)
