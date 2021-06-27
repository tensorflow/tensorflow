# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "alias_with_tflite", "cc_library_with_tflite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

alias_with_tflite(
    name = "native_framework_only",
    actual = ":native_experimental_framework_only",
)

# Native code needed for TF Lite Java API, including experimental APIs;
# but only the framework code, no ops.
cc_library_with_tflite(
    name = "native_experimental_framework_only",
    hdrs = ["jni_utils.h"],
    tflite_deps = [
        ":native_stable_framework_only",
        "//tensorflow/lite/core/shims:framework_experimental",
    ],
    deps = [
        "//tensorflow/lite/java/jni",
    ],
)

# Native code needed for TF Lite Java API, excluding experimental APIs;
# but only the framework code, no ops.
cc_library_with_tflite(
    name = "native_stable_framework_only",
    srcs = [
        "default/jni_utils.cc",
        "jni_utils.cc",
        "nativeinterpreterwrapper_jni.cc",
        "tensor_jni.cc",
        "tensorflow_lite_jni.cc",
    ],
    hdrs = ["jni_utils.h"],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    tflite_deps = [
        "//tensorflow/lite/core/shims:common",
        "//tensorflow/lite/core/shims:framework_stable",
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

alias_with_tflite(
    name = "native",
    actual = ":native_experimental",
)

# Native code and ops needed for the TF Lite Java API, including experimental
# features. This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library_with_tflite(
    name = "native_experimental",
    tflite_deps = [
        ":native_stable",
        "//tensorflow/lite/core/shims:framework_experimental",
    ],
)

# Native code and ops needed for the TF Lite Java API, excluding experimental
# features. This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library_with_tflite(
    name = "native_stable",
    copts = tflite_copts(),
    tflite_deps = [
        ":native_framework_only",
        "//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow/lite/core/shims:framework_stable",
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
