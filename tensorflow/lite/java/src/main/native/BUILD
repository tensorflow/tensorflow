# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

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
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:op_resolver",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/shims:common",
        "//tensorflow/lite/core/shims:framework",
        "//tensorflow/lite/core/shims:jni_initialization",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

# This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library(
    name = "native",
    copts = tflite_copts(),
    deps = [
        ":native_framework_only",
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow/lite/core/shims:framework",
        "//tensorflow/lite/kernels:deprecated_backends",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "exported_symbols.lds",
        "version_script.lds",
    ],
)
