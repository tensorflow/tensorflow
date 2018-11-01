# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

package(default_visibility = ["//visibility:public"])

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "native_framework_only",
    srcs = [
        "exception_jni.cc",
        "nativeinterpreterwrapper_jni.cc",
        "tensor_jni.cc",
        "tensorflow_lite_jni.cc",
    ] + select({
        # The Android toolchain makes "jni.h" available in the include path.
        # For non-Android toolchains, generate jni.h and jni_md.h.
        "//tensorflow:android": [],
        "//conditions:default": [
            ":jni.h",
            ":jni_md.h",
        ],
    }),
    hdrs = [
        "exception_jni.h",
        "nativeinterpreterwrapper_jni.h",
        "tensor_jni.h",
        "tensorflow_lite_jni.h",
    ],
    copts = tflite_copts(),
    includes = select({
        "//tensorflow:android": [],
        "//conditions:default": ["."],
    }),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//tensorflow/lite:context",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
    ],
    alwayslink = 1,
)

# Silly rules to make
# #include <jni.h>
# in the source headers work
# (in combination with the "includes" attribute of the tf_cuda_library rule
# above. Not needed when using the Android toolchain).
#
# Inspired from:
# https://github.com/bazelbuild/bazel/blob/f99a0543f8d97339d32075c7176b79f35be84606/src/main/native/BUILD
# but hopefully there is a simpler alternative to this.
genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = select({
        "//tensorflow:darwin": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)

cc_library(
    name = "init_tensorflow",
    srcs = [
        "init_tensorflow_jni.cc",
    ] + select({
        # The Android toolchain makes "jni.h" available in the include path.
        # For non-Android toolchains, generate jni.h and jni_md.h.
        "//tensorflow:android": [],
        "//conditions:default": [
            ":jni.h",
            ":jni_md.h",
        ],
    }),
    hdrs = [
        "init_tensorflow_jni.h",
    ],
    copts = tflite_copts(),
    includes = select({
        "//tensorflow:android": [],
        "//conditions:default": ["."],
    }),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
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
