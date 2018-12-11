# Description:
# Java Native Interface (JNI) library for testing the TensorFlow Lite Java API.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

cc_library(
    name = "native",
    testonly = 1,
    srcs = [
        "interpreter_test_jni.cc",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/java/jni",
    ],
)

tflite_jni_binary(
    name = "libtensorflowlite_test_jni.so",
    testonly = 1,
    deps = [":native"],
)
