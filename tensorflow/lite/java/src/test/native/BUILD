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
    ] + select({
        # The Android toolchain makes "jni.h" available in the include path.
        # For non-Android toolchains, generate jni.h and jni_md.h.
        "//tensorflow:android": [],
        "//conditions:default": [
            "//tensorflow/lite/java/src/main/native:jni.h",
            "//tensorflow/lite/java/src/main/native:jni_md.h",
        ],
    }),
    includes = select({
        "//tensorflow:android": [],
        "//conditions:default": ["../../main/native/."],
    }),
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

tflite_jni_binary(
    name = "libtensorflowlite_test_jni.so",
    testonly = 1,
    deps = [":native"],
)
