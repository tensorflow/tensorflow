# Description:
# Java Native Interface (JNI) library for testing the TensorFlow Lite Java API.

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "native",
    testonly = 1,
    srcs = [
        "interpreter_test_jni.cc",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

tflite_jni_binary(
    name = "libtensorflowlite_test_jni.so",
    testonly = 1,
    deps = [":native"],
)
