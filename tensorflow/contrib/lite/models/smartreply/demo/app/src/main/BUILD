load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/contrib/lite:build_def.bzl",
    "tflite_copts",
    "tflite_jni_binary",
)

filegroup(
    name = "assets",
    srcs = [
        "@tflite_smartreply//:model_files",
    ],
)

android_binary(
    name = "SmartReplyDemo",
    srcs = glob(["java/**/*.java"]),
    assets = [":assets"],
    assets_dir = "",
    custom_package = "com.example.android.smartreply",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    tags = ["manual"],
    deps = [
        ":smartreply_runtime",
        "@androidsdk//com.android.support:support-v13-25.2.0",
        "@androidsdk//com.android.support:support-v4-25.2.0",
    ],
)

cc_library(
    name = "smartreply_runtime",
    srcs = ["libsmartreply_jni.so"],
    visibility = ["//visibility:public"],
)

tflite_jni_binary(
    name = "libsmartreply_jni.so",
    deps = [
        ":smartreply_jni_lib",
    ],
)

cc_library(
    name = "smartreply_jni_lib",
    srcs = [
        "smartreply_jni.cc",
    ],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/models/smartreply:predictor_lib",
    ],
    alwayslink = 1,
)
