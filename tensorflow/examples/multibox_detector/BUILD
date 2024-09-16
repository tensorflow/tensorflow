# Description:
#   TensorFlow C++ inference example for labeling images.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "detect_objects",
    srcs = [
        "main.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:numbers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)
