# Description:
#   Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "tensorflow_native_libs",
    srcs = glob(["jni/**/*.cc"]),
    hdrs = glob(["jni/**/*.h"]),
    copts = [
        "-std=c++11",
        "-mfpu=neon",
    ],
    linkopts = ["-llog -landroid -lm -ljnigraphics"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":dummy_pthread",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)

# This library only exists as a workaround to satisfy dependencies
# that declare -lpthread in their linkopts. Although Android supports
# pthreads, it does not provide it as a separate library.
cc_library(
    name = "dummy_pthread",
    srcs = ["jni/libpthread.so"],
)

android_binary(
    name = "tensorflow_demo",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "org.tensorflow.demo",
    inline_constants = 1,
    legacy_native_support = 0,
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
    ],
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
