load("@local_tsl//tsl/platform:build_config.bzl", "tsl_cc_test")
load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("@local_tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/tsl/profiler:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "trace_container",
    srcs = ["trace_container.cc"],
    hdrs = ["trace_container.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//xla/tsl/profiler:internal",
    ],
    deps = [
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "xla_op_utils",
    hdrs = ["xla_op_utils.h"],
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//xla/python:__pkg__",
    ]),
    deps = ["@com_google_absl//absl/strings"],
)

tsl_cc_test(
    name = "xla_op_utils_test",
    size = "small",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "post_process_single_host_xplane",
    srcs = ["post_process_single_host_xplane.cc"],
    hdrs = ["post_process_single_host_xplane.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility(["//xla/tsl/profiler:internal"]),
    deps = [
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:timestamp_utils",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        ":trace_container",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@local_tsl//tsl/profiler/utils:format_utils",
        "@local_tsl//tsl/profiler/utils:math_utils",
    ],
)

tsl_cc_test(
    name = "trace_container_test",
    srcs = ["trace_container_test.cc"],
    deps = [
        ":trace_container",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

tsl_cc_test(
    name = "trace_events_to_json_test",
    srcs = ["trace_events_to_json_test.cc"],
    deps = [
        ":trace_container",
        ":trace_events_to_json",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "xplane_to_trace_events",
    srcs = ["xplane_to_trace_events.cc"],
    hdrs = ["xplane_to_trace_events.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        ":trace_container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@local_tsl//tsl/profiler/utils:trace_utils",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
        "@local_tsl//tsl/profiler/utils:xplane_visitor",
    ],
)

tsl_cc_test(
    name = "xplane_to_trace_events_test",
    size = "small",
    srcs = ["xplane_to_trace_events_test.cc"],
    deps = [
        ":xplane_to_trace_events",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:trace_utils",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
    ],
)
