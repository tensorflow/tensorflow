# Description:
#   BenchmarkModel Android harness for Firebase Test Lab.

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_hexagon_nn_skel_libraries")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

android_binary(
    name = "benchmark_model_firebase",
    srcs = glob([
        "src/**/*.java",
    ]),
    custom_package = "org.tensorflow.lite.benchmark.firebase",
    manifest = "AndroidManifest.xml",
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        ":hexagon_libs",
        ":tensorflowlite_benchmark_firebase_native",
    ],
)

tflite_jni_binary(
    name = "libtensorflowlite_benchmark_firebase.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    deps = [
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)

cc_library(
    name = "tensorflowlite_benchmark_firebase_native",
    srcs = ["libtensorflowlite_benchmark_firebase.so"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "hexagon_libs",
    srcs = select({
        "//tensorflow:android_arm64": [
            "//tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "//tensorflow:android_arm": [
            "//tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "//conditions:default": [],
    }),
    visibility = ["//visibility:private"],
)
