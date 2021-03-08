load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "pluggable_device_plugin_init",
    srcs = [
        "pluggable_device_plugin_init.cc",
    ],
    hdrs = [
        "pluggable_device_plugin_init.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c/experimental/grappler",
    ],
)
