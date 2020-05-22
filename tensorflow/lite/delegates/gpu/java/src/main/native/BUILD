# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite GPU delegate Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "native",
    srcs = ["gpu_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)
