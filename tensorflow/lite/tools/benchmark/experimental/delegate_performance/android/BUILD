# Description:
#   Delegate Performance Benchmark (DPB) Android app.
#   This provides model-level latency & accuracy testings for delegates, on Android.

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

android_binary(
    name = "delegate_performance_benchmark",
    srcs = glob([
        "src/**/*.java",
    ]),
    custom_package = "org.tensorflow.lite.benchmark.delegate_performance",
    manifest = "AndroidManifest.xml",
    multidex = "native",
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflowlite_delegate_performance_benchmark_native",
    ],
)

tflite_jni_binary(
    name = "libtensorflowlite_delegate_performance_benchmark.so",
    srcs = ["jni/delegate_performance_benchmark_jni.cc"],
    deps = [
        ":latency_benchmark",
        "//tensorflow/lite/java/jni",
    ],
)

cc_library(
    name = "latency_benchmark",
    srcs = ["jni/latency_benchmark.cc"],
    hdrs = ["jni/latency_benchmark.h"],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)

cc_library(
    name = "tensorflowlite_delegate_performance_benchmark_native",
    srcs = ["libtensorflowlite_delegate_performance_benchmark.so"],
)
