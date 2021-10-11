# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite NNAPI Delegate Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "nnapi_native_srcs_visibility_allowlist")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "native",
    srcs = ["nnapi_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

filegroup(
    name = "native_srcs",
    srcs = ["nnapi_delegate_jni.cc"],
    visibility = nnapi_native_srcs_visibility_allowlist(),
)
