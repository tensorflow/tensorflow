# Description:
#   TensorFlow C++ inference example for labeling images.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(licenses = ["notice"])

tf_cc_binary(
    name = "detect_objects",
    srcs = [
        "main.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)
