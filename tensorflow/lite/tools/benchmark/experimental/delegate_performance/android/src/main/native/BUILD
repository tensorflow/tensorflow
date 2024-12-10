# Description:
#  Holds the native layer of the app.

load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "jni_binary_with_tflite")
load("//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:build_defs.bzl", "accuracy_benchmark_extra_deps", "latency_benchmark_extra_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

jni_binary_with_tflite(
    name = "libdelegate_performance_benchmark.so",
    srcs = ["delegate_performance_benchmark_jni.cc"],
    tflite_deps = [":accuracy_benchmark"],
    deps = [
        ":latency_benchmark",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@flatbuffers",
    ],
)

cc_library(
    name = "status_codes",
    hdrs = ["status_codes.h"],
)

cc_library(
    name = "latency_benchmark",
    srcs = ["latency_benchmark.cc"],
    hdrs = ["latency_benchmark.h"],
    deps = [
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/profiling:memory_info",
        "//tensorflow/lite/tools/benchmark:benchmark_model_lib",
        "//tensorflow/lite/tools/benchmark:benchmark_params",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + latency_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "accuracy_benchmark",
    srcs = ["accuracy_benchmark.cc"],
    hdrs = ["accuracy_benchmark.h"],
    deps = [
        ":status_codes",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/acceleration/configuration:gpu_plugin",
        "//tensorflow/lite/acceleration/configuration:stable_delegate_plugin",
        "//tensorflow/lite/acceleration/configuration:xnnpack_plugin",
        "//tensorflow/lite/core/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:libjpeg_handle_static_link",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark/c:c_api_in_process",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "@flatbuffers",
    ] + accuracy_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "benchmark_native",
    tflite_jni_binaries = [":libdelegate_performance_benchmark.so"],
)
