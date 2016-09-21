# Description:
#   JNI-based Java inference interface for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "if_android",
)

# TODO(andrewharp): Make this an android_library or java_library.
filegroup(
    name = "android_tensorflow_inference_java_srcs",
    srcs = glob(["java/**/*.java"]),
    visibility = ["//visibility:public"],
)

exports_files([
    "jni/version_script.lds",
])

filegroup(
    name = "android_tensorflow_inference_jni_srcs",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "android_tensorflow_inference_jni",
    srcs = if_android([":android_tensorflow_inference_jni_srcs"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:android_tensorflow_lib_lite",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
