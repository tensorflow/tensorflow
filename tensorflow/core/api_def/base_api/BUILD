# Description:
#   Expose TensorFlow base api.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
