# Description:
#   Delegate Performance Benchmark (DPB) Android app.
#   This provides model-level latency & accuracy testings for delegates, on Android.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "android_binary_with_tflite", "android_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

android_library(
    name = "benchmark_accuracy_activity",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkAccuracyActivity.java"],
    deps = [":benchmark_accuracy_impl"],
)

android_library(
    name = "benchmark_accuracy_impl",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkAccuracyImpl.java"],
    deps = [
        ":benchmark_accuracy",
        ":benchmark_report",
        ":csv_writer",
        ":delegate_performance_benchmark_utils",
        ":html_writer",
        ":json_writer",
        ":model_benchmark_report",
        ":raw_delegate_metrics_entry",
        ":tflite_settings_list_entry",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs_android",
    ],
)

android_library(
    name = "benchmark_accuracy",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkAccuracy.java"],
)

android_library(
    name = "benchmark_latency_activity",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkLatencyActivity.java"],
    deps = [":benchmark_latency_impl"],
)

android_library(
    name = "benchmark_latency_impl",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkLatencyImpl.java"],
    deps = [
        ":benchmark_report",
        ":csv_writer",
        ":delegate_performance_benchmark_utils",
        ":html_writer",
        ":json_writer",
        ":model_benchmark_report",
        ":model_benchmark_report_interface",
        ":preconditions",
        ":raw_delegate_metrics_entry",
        ":tflite_settings_list_entry",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_java_proto_lite",
    ],
)

android_library_with_tflite(
    name = "benchmark_report",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkReport.java",
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/ReportWriter.java",
    ],
    deps = [
        ":benchmark_result_type",
        ":delegate_performance_benchmark_utils",
        ":model_benchmark_report_interface",
    ],
)

android_library(
    name = "benchmark_result_type",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkResultType.java"],
)

android_library_with_tflite(
    name = "csv_writer",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/CsvWriter.java"],
    tflite_deps = [":benchmark_report"],
    deps = [
        ":delegate_metrics_entry",
        ":metrics_entry",
        ":model_benchmark_report_interface",
        ":preconditions",
    ],
)

android_library(
    name = "delegate_metrics_entry",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/DelegateMetricsEntry.java"],
    deps = [
        ":benchmark_result_type",
        ":metrics_entry",
    ],
)

android_library_with_tflite(
    name = "delegate_performance_benchmark_lib",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkAccuracyActivity.java",
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkLatencyActivity.java",
    ],
    tflite_deps = [
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native:benchmark_native",
    ],
    deps = [
        ":benchmark_accuracy_impl",
        ":benchmark_latency_impl",
    ],
)

android_library(
    name = "delegate_performance_benchmark_utils",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/DelegatePerformanceBenchmark.java",
    ],
    deps = [
        ":benchmark_result_type",
        ":preconditions",
        ":tflite_settings_list_entry",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs_android",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_java_proto_lite",
        "@flatbuffers//:runtime_android",
    ],
)

android_library_with_tflite(
    name = "html_writer",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/HtmlWriter.java"],
    tflite_deps = [":benchmark_report"],
    deps = [
        ":benchmark_result_type",
        ":delegate_metrics_entry",
        ":metrics_entry",
        ":model_benchmark_report_interface",
        ":preconditions",
    ],
)

android_library_with_tflite(
    name = "json_writer",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/JsonWriter.java"],
    tflite_deps = [":benchmark_report"],
)

android_library(
    name = "metrics_entry",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/MetricsEntry.java"],
    deps = [":benchmark_result_type"],
)

android_library_with_tflite(
    name = "model_benchmark_report",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/AccuracyBenchmarkReport.java",
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/LatencyBenchmarkReport.java",
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/ModelBenchmarkReport.java",
    ],
    deps = [
        ":benchmark_result_type",
        ":delegate_metrics_entry",
        ":delegate_performance_benchmark_utils",
        ":metrics_entry",
        ":model_benchmark_report_interface",
        ":preconditions",
        ":raw_delegate_metrics_entry",
        ":tflite_settings_list_entry",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs_android",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_java_proto_lite",
    ],
)

android_library(
    name = "model_benchmark_report_interface",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/ModelBenchmarkReportInterface.java",
    ],
    deps = [
        ":benchmark_result_type",
        ":delegate_metrics_entry",
    ],
)

android_library(
    name = "preconditions",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/Preconditions.java"],
)

android_library(
    name = "raw_delegate_metrics_entry",
    srcs = ["src/main/java/org/tensorflow/lite/benchmark/delegateperformance/RawDelegateMetricsEntry.java"],
    deps = ["//tensorflow/lite/acceleration/configuration:configuration_fbs_android"],
)

android_library(
    name = "tflite_settings_list_entry",
    srcs = [
        "src/main/java/org/tensorflow/lite/benchmark/delegateperformance/TfLiteSettingsListEntry.java",
    ],
    deps = [
        ":preconditions",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs_android",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_java_proto_lite",
    ],
)

# The main test app.
android_binary_with_tflite(
    name = "delegate_performance_benchmark",
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
    tflite_deps = [
        ":delegate_performance_benchmark_lib",
    ],
    visibility = ["//visibility:public"],
)
