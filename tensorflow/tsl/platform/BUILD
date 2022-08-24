# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "stack_frame",
    hdrs = ["stack_frame.h"],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "stack_frame.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)
