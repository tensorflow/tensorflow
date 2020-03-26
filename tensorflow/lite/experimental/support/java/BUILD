# Description:
# TensorFlow Lite Support API in Java.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(138904786): Split Java part and Android part to make the support library usable by pure Java.
android_library(
    name = "tensorflow-lite-support",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java:tensorflowlite_gpu",
        "@org_checkerframework_qual",
    ],
)

# This alias matches the style of lite/java naming for android_library targets. We keep the
# `tensorflow-lite-support` variant to match the associated .aar library name output style.
alias(
    name = "tensorflowlite_support",
    actual = ":tensorflow-lite-support",
)

java_library(
    name = "tensorflow-lite-support-precondition",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "tensorflow-lite-support-precondition-lib-android",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "@org_checkerframework_qual",
    ],
)
