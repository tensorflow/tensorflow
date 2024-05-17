load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//tsl/profiler:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "trace_container",
    srcs = ["trace_container.cc"],
    hdrs = ["trace_container.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tsl/profiler:internal",
    ],
    deps = [
        "//tsl/platform:protobuf",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "xla_op_utils",
    hdrs = ["xla_op_utils.h"],
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_profiler_backends",
        "@local_xla//xla/python:__pkg__",
    ]),
    deps = ["@com_google_absl//absl/strings"],
)

tsl_cc_test(
    name = "xla_op_utils_test",
    size = "small",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "post_process_single_host_xplane",
    srcs = ["post_process_single_host_xplane.cc"],
    hdrs = ["post_process_single_host_xplane.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility(["//tsl/profiler:internal"]),
    deps = [
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:timestamp_utils",
        "//tsl/profiler/utils:xplane_schema",
        "//tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    deps = [
        ":trace_container",
        "//tsl/platform:protobuf",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
        "//tsl/profiler/utils:format_utils",
        "//tsl/profiler/utils:math_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ],
)

tsl_cc_test(
    name = "trace_container_test",
    srcs = ["trace_container_test.cc"],
    deps = [
        ":trace_container",
        "//tsl/platform:protobuf",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

tsl_cc_test(
    name = "trace_events_to_json_test",
    srcs = ["trace_events_to_json_test.cc"],
    deps = [
        ":trace_container",
        ":trace_events_to_json",
        "//tsl/platform:protobuf",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "xplane_to_trace_events",
    srcs = ["xplane_to_trace_events.cc"],
    hdrs = ["xplane_to_trace_events.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    deps = [
        ":trace_container",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:tf_xplane_visitor",
        "//tsl/profiler/utils:trace_utils",
        "//tsl/profiler/utils:xplane_schema",
        "//tsl/profiler/utils:xplane_utils",
        "//tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tsl_cc_test(
    name = "xplane_to_trace_events_test",
    size = "small",
    srcs = ["xplane_to_trace_events_test.cc"],
    deps = [
        ":xplane_to_trace_events",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:trace_events_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:trace_utils",
        "//tsl/profiler/utils:xplane_builder",
        "//tsl/profiler/utils:xplane_schema",
    ],
)
