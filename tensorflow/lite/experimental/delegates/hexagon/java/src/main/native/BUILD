# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Hexagon delegate Java API using the TensorFlow Lite CC library.

package(default_visibility = ["//tensorflow/lite/experimental/delegates/hexagon/java:__subpackages__"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "native",
    srcs = ["hexagon_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)
