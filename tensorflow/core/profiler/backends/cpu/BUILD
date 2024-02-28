load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "traceme_recorder",
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/profiler/backends/cpu:traceme_recorder",
    ] + if_static([
        ":traceme_recorder_impl",
    ]),
)

cc_library(
    name = "traceme_recorder_impl",
    hdrs = [
        "traceme_recorder.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer",
    deps = [
        "@local_xla//xla/backends/profiler/cpu:python_tracer",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer_impl",
    hdrs = ["python_tracer.h"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_xla//xla/backends/profiler/cpu:python_tracer_impl",
    ],
)

cc_library(
    name = "metadata_collector",
    copts = tf_profiler_copts(),
    deps = [
        "@local_xla//xla/backends/profiler/cpu:metadata_collector",
    ],
    alwayslink = True,
)
