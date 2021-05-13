# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Flex delegate for using TensorFlow ops with TensorFlow Lite.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(
    srcs = ["flex_delegate_jni.cc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "native",
    srcs = ["flex_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/testing:init_tensorflow",
    ],
    alwayslink = 1,
)
