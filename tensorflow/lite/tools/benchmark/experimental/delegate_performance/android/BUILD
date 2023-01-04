# Description:
#   Delegate Performance Benchmark (DPB) Android app.
#   This provides model-level latency & accuracy testings for delegates, on Android.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# The main test app.
android_binary(
    name = "delegate_performance_benchmark",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = [
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/models:accuracy_models",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/models:latency_criteria_files",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/models:latency_models",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:default_latency_criteria",
    ],
    assets_dir = "assets",
    custom_package = "org.tensorflow.lite.benchmark.delegate_performance",
    manifest = "AndroidManifest.xml",
    multidex = "native",
    nocompress_extensions = [".tflite"],
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs_android",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni:tensorflowlite_delegate_performance_benchmark_native",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_java_proto_lite",
        "@flatbuffers//:runtime_android",
    ],
)
