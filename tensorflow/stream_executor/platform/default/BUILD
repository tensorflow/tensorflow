load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_copts")

licenses(["notice"])

package(default_visibility = ["//tensorflow/stream_executor:__subpackages__"])

cc_library(
    name = "platform",
    textual_hdrs = ["initialize.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform/default:platform",
    ],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tf_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform/default:dso_loader",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)
