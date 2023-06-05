# Description:
# Platform-specific build configurations.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_copts")

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "platformlib",
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/platform/default/build_config:gif",
        "//tensorflow/tsl/platform/default/build_config:jpeg",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d",
        "@highwayhash//:sip_hash",
        "@zlib",
    ],
)
