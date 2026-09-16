load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tfl_calibration_utils",
    srcs = ["tfl_calibration_utils.cc"],
    hdrs = ["tfl_calibration_utils.h"],
    deps = [
        ":tfl_tensor_stats_profiler",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:stateful_error_reporter",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:memory_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "tfl_tensor_stats_profiler",
    srcs = ["tfl_tensor_stats_profiler.cc"],
    hdrs = ["tfl_tensor_stats_profiler.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:private_common",
    ],
)

cc_test(
    name = "tfl_tensor_stats_profiler_test",
    srcs = ["tfl_tensor_stats_profiler_test.cc"],
    deps = [
        ":tfl_tensor_stats_profiler",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tfl_calibration_utils_test",
    srcs = ["tfl_calibration_utils_test.cc"],
    tags = ["nohwasan"],
    deps = [
        ":tfl_calibration_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/profiling:memory_info",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "_pywrap_tfl_calibration",
    srcs = ["tfl_calibration_wrapper.cc"],
    module_name = "_pywrap_tfl_calibration",
    wrap_py_init = True,
    deps = [
        ":tfl_calibration_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:stateful_error_reporter",
        "//tensorflow/lite/profiling:memory_info",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)
