load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//perftools/gputools/profiler/collector:__pkg__",
        "//tensorflow/core/profiler:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "trace_events_filter_interface",
    hdrs = ["trace_events_filter_interface.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "trace_viewer_visibility",
    srcs = ["trace_viewer_visibility.cc"],
    hdrs = ["trace_viewer_visibility.h"],
    deps = [
        ":trace_events_filter_interface",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/utils:timespan",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "trace_viewer_visibility_test",
    srcs = ["trace_viewer_visibility_test.cc"],
    deps = [
        ":trace_viewer_visibility",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/utils:timespan",
    ],
)

cc_library(
    name = "trace_viewer_color",
    srcs = ["trace_viewer_color.cc"],
    hdrs = ["trace_viewer_color.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    deps = [
        ":trace_events_util",
        ":trace_viewer_color",
        "//tensorflow/core/profiler/lib:context_types",
        "//tensorflow/core/profiler/protobuf:task_proto_cc",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/protobuf:trace_events_raw_proto_cc",
        "//tensorflow/core/profiler/utils:timespan",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "trace_event_arguments_builder",
    hdrs = ["trace_event_arguments_builder.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:trace_events_raw_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_events_util",
    srcs = ["trace_events_util.cc"],
    hdrs = ["trace_events_util.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/utils:timespan",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "trace_events",
    srcs = ["trace_events.cc"],
    hdrs = ["trace_events.h"],
    deps = [
        ":trace_events_filter_interface",
        ":trace_events_util",
        ":trace_viewer_visibility",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:context_types_hdrs",
        "//tensorflow/core/profiler/protobuf:task_proto_cc",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/protobuf:trace_events_raw_proto_cc",
        "//tensorflow/core/profiler/utils:timespan",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
