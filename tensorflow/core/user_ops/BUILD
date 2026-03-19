# User ops.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
    ],
    alwayslink = 1,
)
