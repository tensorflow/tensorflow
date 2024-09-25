# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Hexagon delegate Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/delegates/hexagon/java:__subpackages__"],
)

licenses(["notice"])

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
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)
