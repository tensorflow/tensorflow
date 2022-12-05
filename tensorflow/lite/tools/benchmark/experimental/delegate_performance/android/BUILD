# Description:
#   Delegate Performance Benchmark (DPB) Android app.
#   This provides model-level latency & accuracy testings for delegates, on Android.

load("//tensorflow/lite/experimental/acceleration/mini_benchmark:build_defs.bzl", "validation_model")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow:tensorflow.bzl", "clean_dep")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

android_binary(
    name = "delegate_performance_benchmark",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = [":mobilenet_float_with_validation.tflite"],
    assets_dir = "",
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

validation_model(
    name = "mobilenet_float_with_validation.tflite",
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:odt_classifier_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224.tflite",
    metrics_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/metrics:mobilenet_metrics_tflite",
)

tflite_jni_binary(
    name = "libtensorflowlite_delegate_performance_benchmark.so",
    srcs = ["jni/delegate_performance_benchmark_jni.cc"],
    deps = [
        ":accuracy_benchmark",
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
    name = "accuracy_benchmark",
    srcs = ["jni/accuracy_benchmark.cc"],
    hdrs = ["jni/accuracy_benchmark.h"],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "@flatbuffers//:public_headers_lib",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:blocking_validator_runner",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:validator_runner_entrypoint",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:validator_runner_options",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration:stable_delegate_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration:xnnpack_plugin",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
    ] + select({
        # On Android, as the validation runs in a separate process as a
        # different binary, any TFLite delegates to be validated need to
        # include corresponding delegate plugins.
        clean_dep("//tensorflow:android"): [
            "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "accuracy_benchmark_test",
    srcs = ["jni/accuracy_benchmark_test.cc"],
    data = [
        "//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:test_xnnpack_settings.json",
        "//tensorflow/lite/tools/delegates/experimental/stable_delegate:test_invalid_settings.json",
        "//tensorflow/lite/tools/delegates/experimental/stable_delegate:test_sample_stable_delegate_settings.json",
    ],
    deps = [
        ":accuracy_benchmark",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:embedded_mobilenet_validation_model",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensorflowlite_delegate_performance_benchmark_native",
    srcs = ["libtensorflowlite_delegate_performance_benchmark.so"],
)
