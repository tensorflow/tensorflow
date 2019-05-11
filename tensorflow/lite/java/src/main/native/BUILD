# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

package(default_visibility = ["//visibility:public"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "native_framework_only",
    srcs = [
        "jni_utils.cc",
        "nativeinterpreterwrapper_jni.cc",
        "tensor_jni.cc",
        "tensorflow_lite_jni.cc",
    ],
    hdrs = [
        "jni_utils.h",
    ],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

cc_library(
    name = "init_tensorflow",
    srcs = [
        "init_tensorflow_jni.cc",
    ],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/testing:init_tensorflow",
    ],
    alwayslink = 1,
)

# This includes all ops. If you want a smaller binary, you should copy and
# modify builtin_ops_jni.cc.  You should then link your binary against both
# ":native_framework_only" and your own version of ":native_builtin_ops".
cc_library(
    name = "native",
    srcs = [
        "builtin_ops_jni.cc",
    ],
    copts = tflite_copts(),
    deps = [
        ":native_framework_only",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "version_script.lds",
    ],
)
