# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite GPU delegate Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(srcs = ["gpu_delegate_jni.cc"])

cc_library_with_tflite(
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
    tflite_deps = [
        "//tensorflow/lite/java/src/main/native:jni_utils",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:request_gpu_info",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite/experimental/acceleration/compatibility:gpu_compatibility",
        "//tensorflow/lite/java/jni",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
