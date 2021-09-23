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
    srcs = [
        ":native_experimental_only_srcs",
    ],
    hdrs = [":native_hdrs"],
    tflite_deps = [
        ":native_stable_framework_only",
        "//tensorflow/lite/core/shims:framework_experimental",
    ],
    deps = [
        "//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

filegroup(
    name = "native_experimental_only_srcs",
    srcs = [
        "nativeinterpreterwrapperexperimental_jni.cc",
    ],
)

filegroup(
    name = "native_hdrs",
    srcs = ["jni_utils.h"],
)

# Native code needed for TF Lite Java API, excluding experimental APIs;
# but only the framework code, no ops.
filegroup(
    name = "native_stable_srcs",
    srcs = [
        "jni_utils.cc",
        "nativeinterpreterwrapper_jni.cc",
        "nativesignaturerunner_jni.cc",
        "tensor_jni.cc",
        "tensorflow_lite_jni.cc",
        "xnnpackdelegate_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "native_stable_framework_only",
    srcs = [":native_stable_srcs"],
    hdrs = [":native_hdrs"],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    tflite_deps = [
        "//tensorflow/lite/core/shims:common",
        "//tensorflow/lite/core/shims:create_op_resolver_header",
        "//tensorflow/lite/core/shims:framework_stable",
        "//tensorflow/lite/core/shims:jni_initialization",
        "//tensorflow/lite/core/shims:jni_utils",
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
    copts = tflite_copts(),
    tflite_deps = [
        ":native_experimental_framework_only",
        "//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow/lite/core/shims:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/core/shims:framework",
    ],
    deps = [
        "//tensorflow/lite/core/api",
    ],
    alwayslink = 1,
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
        "//tensorflow/lite/core/shims:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/core/shims:framework_stable",
    ],
    deps = [
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
