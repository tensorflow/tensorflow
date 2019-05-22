# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite GPU delegate Java API using the TensorFlow Lite CC library.

package(default_visibility = ["//visibility:public"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "native",
    srcs = ["gpu_delegate_jni.cc"],
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-lGLESv3",
            "-lEGL",
        ],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu:gl_delegate",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)
