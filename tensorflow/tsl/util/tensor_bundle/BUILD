load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "byte_swap_array",
    srcs = ["byte_swap_array.cc"],
    hdrs = ["byte_swap_array.h"],
    deps = [
        "//tensorflow/tsl/platform:byte_order",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
    ],
)

filegroup(
    name = "framework_internal_private_hdrs",
    srcs = [
        "byte_swap_array.h",
    ],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "byte_swap_array.cc",
        "byte_swap_array.h",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "byte_swap_array.cc",
        "byte_swap_array.h",
    ],
)
