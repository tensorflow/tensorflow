# Description:
#   Expose TensorFlow base api.

load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "base_api_def",
    srcs = glob(
        [
            "*",
        ],
        exclude = [
            "BUILD",
        ],
    ),
    visibility = ["//tensorflow:internal"],
)
