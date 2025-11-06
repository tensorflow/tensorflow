load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "ondevice_trace_event",
    hdrs = ["ondevice_trace_event.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "ondevice_event_collector",
    hdrs = ["ondevice_event_collector.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        ":ondevice_trace_event",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "ondevice_event_receiver",
    srcs = ["ondevice_event_receiver.cc"],
    hdrs = ["ondevice_event_receiver.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        ":ondevice_event_collector",
        ":ondevice_trace_event",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)
