load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# EXPERIMENTAL: Native target that runs inference on the Hexagon backend.
# The Hexagon backend-related targets are intentionally not included in BUILD.bazel.
tflite_jni_binary(
    name = "libtensorflowlite_hexagon_jni.so",
    linkscript = "//tensorflow/lite/experimental/delegates/hexagon:version_script.lds",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/experimental/delegates/hexagon/java/src/main/native",
    ],
)

cc_library(
    name = "tensorflowlite_hexagon",
    srcs = [
        "libtensorflowlite_hexagon_jni.so",
    ] + select({
        "//tensorflow:android_arm64": ["@hexagon_nn//:hexagon/arm64-v8a/libhexagon_interface.so"],
        "//tensorflow:android_arm": ["@hexagon_nn//:hexagon/armeabi-v7a/libhexagon_interface.so"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
)

android_library(
    name = "tensorflowlite_java_hexagon",
    srcs = ["//tensorflow/lite/experimental/delegates/hexagon/java/src/main/java/org/tensorflow/lite/experimental:hexagon_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":tensorflowlite_hexagon",
        "//tensorflow/lite/java:tensorflowlite_java",
        "@org_checkerframework_qual",
    ],
)

aar_with_jni(
    name = "tensorflow-lite-hexagon",
    android_library = ":tensorflowlite_java_hexagon",
    headers = ["//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate.h"],
)
