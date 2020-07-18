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
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tensorflow/lite/experimental/tflite_api_dispatcher:tflite_api_dispatcher_with_kernels",
        "//tensorflow/lite/java/jni",
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
    hdrs = ["op_resolver.h"],
    copts = tflite_copts(),
    deps = [
        ":native_framework_only",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

# TODO(b/153652701): Generate this target to give CreateOpResolver a custom namespace.
cc_library(
    name = "selected_ops_jni",
    srcs = ["selected_ops_jni.cc"],
    hdrs = ["op_resolver.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
    ],
)

exports_files(
    [
        "exported_symbols.lds",
        "version_script.lds",
    ],
)
