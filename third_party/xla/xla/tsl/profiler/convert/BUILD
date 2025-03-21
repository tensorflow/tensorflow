load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_alias", "tf_profiler_copts")

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
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "xla_op_utils",
    srcs = [tf_profiler_alias("//xla/tsl/profiler/convert/", "xla_op_utils.cc")],
    hdrs = ["xla_op_utils.h"],
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//xla/python:__pkg__",
    ]),
    deps = [
        "//xla/tsl/platform:macros",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

tsl_cc_test(
    name = "xla_op_utils_test",
    size = "small",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "post_process_single_host_xplane",
    srcs = ["post_process_single_host_xplane.cc"],
    hdrs = ["post_process_single_host_xplane.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility(["//xla/tsl/profiler:internal"]),
    deps = [
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/utils:timestamp_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/utils:format_utils",
        "//xla/tsl/profiler/utils:math_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
    ],
)

tsl_cc_test(
    name = "trace_container_test",
    srcs = ["trace_container_test.cc"],
    deps = [
        ":trace_container",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

tsl_cc_test(
    name = "trace_events_to_json_test",
    srcs = ["trace_events_to_json_test.cc"],
    deps = [
        ":trace_container",
        ":trace_events_to_json",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:protobuf",
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
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "xplane_to_trace_events_test",
    size = "small",
    srcs = ["xplane_to_trace_events_test.cc"],
    deps = [
        ":xplane_to_trace_events",
        "//xla/tsl/platform:test",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
