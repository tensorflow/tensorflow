# Description:
# gif test data packages.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "gif_testdata",
    srcs = [
        # GIF data
        "lena.gif",
        "scan.gif",
        "red_black.gif",
        "squares.gif",
        "3g_multiframe.gif",
        "pendulum_sm.gif",
        # Add groundtruth frames for `pendulum_sm.gif`.
        # PNG format because it's lossless.
        "pendulum_sm_frame0.png",
        "pendulum_sm_frame1.png",
        "pendulum_sm_frame2.png",
        # GIF data with optimization
        "optimized.gif",
    ],
    visibility = ["//visibility:public"],
)
