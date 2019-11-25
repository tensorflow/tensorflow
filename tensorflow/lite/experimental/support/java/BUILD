# Description:
# TensorFlow Lite Support API in Java.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("@build_bazel_rules_android//android:rules.bzl", "android_library")

# TODO(138904786): Split Java part and Android part to make the support library usable by pure Java.
android_library(
    name = "tensorflow-lite-support",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
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
