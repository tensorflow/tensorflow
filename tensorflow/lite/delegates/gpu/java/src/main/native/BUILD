# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite GPU delegate Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "gpu_delegate_jni_srcs",
    srcs = [
        "gpu_delegate_jni.cc",
        "gpu_delegate_native_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "compatibility_list_jni",
    srcs = ["compatibility_list_jni.cc"],
    copts = tflite_copts(),
    tflite_deps = [
        "//tensorflow/lite/java/src/main/native:jni_utils",
    ],
    deps = [
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

cc_library_with_tflite(
    name = "gpu_delegate_jni",
    srcs = ["gpu_delegate_jni.cc"],
    copts = tflite_copts(),
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

cc_library(
    name = "gpu_delegate_native_jni",
    srcs = ["gpu_delegate_native_jni.cc"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "native",
    linkopts = select({
        "//tensorflow:android": [
            "-lGLESv3",
            "-lEGL",
        ],
        "//conditions:default": [],
    }),
    tflite_deps = [
        ":compatibility_list_jni",
        ":gpu_delegate_jni",
    ],
    deps = [
        ":gpu_delegate_native_jni",
    ],
)
