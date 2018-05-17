# Description:
# Internal helper function to test TF Lite API.

load("//tools/build_defs/android:rules.bzl", "android_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

android_library(
    name = "testhelper",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tensorflow/contrib/lite/java:tensorflowlite_java",
    ],
)
