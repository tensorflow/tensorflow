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
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:request_gpu_info",
        "//tensorflow/lite/experimental/acceleration/whitelist:android_info",
        "//tensorflow/lite/experimental/acceleration/whitelist:gpu_whitelist",
        "//tensorflow/lite/java/jni",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
