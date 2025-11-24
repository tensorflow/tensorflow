# Description:
#   TensorFlow camera demo app for Android.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

LINKER_SCRIPT = "jni/version_script.lds"

# libtensorflow_demo.so contains the native code for image colorspace conversion
# and object tracking used by the demo. It does not require TF as a dependency
# to build if STANDALONE_DEMO_LIB is defined.
# TF support for the demo is provided separately by libtensorflow_inference.so.
cc_binary(
    name = "libtensorflow_demo.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    copts = tf_copts(),
    defines = ["STANDALONE_DEMO_LIB"],
    linkopts = [
        "-landroid",
        "-ldl",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--version-script,$(location {})".format(LINKER_SCRIPT),
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        LINKER_SCRIPT,
    ],
)

cc_library(
    name = "tensorflow_native_libs",
    srcs = [
        ":libtensorflow_demo.so",
        "//tensorflow/tools/android/inference_interface:libtensorflow_inference.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

android_binary(
    name = "tensorflow_demo",
    srcs = glob([
        "src/**/*.java",
    ]),
    # Package assets from assets dir as well as all model targets. Remove undesired models
    # (and corresponding Activities in source) to reduce APK size.
    assets = [
        "//tensorflow/tools/android/test/assets:asset_files",
        ":external_assets",
    ],
    assets_dir = "",
    custom_package = "org.tensorflow.demo",
    manifest = "AndroidManifest.xml",
    multidex = "native",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
        "//tensorflow/tools/android/inference_interface:android_tensorflow_inference_java",
    ],
)

# LINT.IfChange
filegroup(
    name = "external_assets",
    srcs = [
        "@inception_v1//:model_files",
        "@mobile_ssd//:model_files",
        "@speech_commands//:model_files",
        "@stylize//:model_files",
    ],
)
# LINT.ThenChange(//tensorflow/tools/android/test/download-models.gradle)

filegroup(
    name = "java_files",
    srcs = glob(["src/**/*.java"]),
)

filegroup(
    name = "jni_files",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
)

filegroup(
    name = "resource_files",
    srcs = glob(["res/**"]),
)

exports_files([
    "AndroidManifest.xml",
])
