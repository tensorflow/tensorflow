# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
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
    tflite_deps = [
        "//tensorflow/lite/core/shims:common",
        "//tensorflow/lite/core/shims:framework",
        "//tensorflow/lite/core/shims:jni_initialization",
    ],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:op_resolver",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

# This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library_with_tflite(
    name = "native",
    copts = tflite_copts(),
    tflite_deps = [
        ":native_framework_only",
        "//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow/lite/core/shims:framework",
    ],
    deps = [
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/core/api",
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
