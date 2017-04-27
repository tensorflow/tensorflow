# Description:
#   TensorFlow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_opts_nortti_if_android",
)

exports_files(["LICENSE"])

LINKER_SCRIPT = "//tensorflow/contrib/android:jni/version_script.lds"

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
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--version-script",  # This line must be directly followed by LINKER_SCRIPT.
        LINKER_SCRIPT,
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
        "//tensorflow/contrib/android:libtensorflow_inference.so",
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
        "//tensorflow/examples/android/assets:asset_files",
        ":external_assets",
    ],
    assets_dir = "",
    custom_package = "org.tensorflow.demo",
    inline_constants = 1,
    manifest = "AndroidManifest.xml",
    manifest_merger = "legacy",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
        "//tensorflow/contrib/android:android_tensorflow_inference_java",
    ],
)

# LINT.IfChange
filegroup(
    name = "external_assets",
    srcs = [
        "@inception5h//:model_files",
        "@mobile_multibox//:model_files",
        "@stylize//:model_files",
    ],
)
# LINT.ThenChange(//tensorflow/examples/android/download-models.gradle)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
            "gradleBuild/**",
            "libs/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

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

exports_files(["AndroidManifest.xml"])
