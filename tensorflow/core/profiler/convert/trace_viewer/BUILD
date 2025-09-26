load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "trace_events_filter_interface",
    hdrs = ["trace_events_filter_interface.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_filter_interface",
    ],
)

cc_library(
    name = "trace_viewer_visibility",
    hdrs = ["trace_viewer_visibility.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_viewer_visibility",
    ],
)

cc_library(
    name = "trace_viewer_color",
    hdrs = ["trace_viewer_color.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_viewer_color",
    ],
)

cc_library(
    name = "trace_events_to_json",
    hdrs = ["trace_events_to_json.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_to_json",
    ],
)

cc_library(
    name = "trace_event_arguments_builder",
    hdrs = ["trace_event_arguments_builder.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_event_arguments_builder",
    ],
)

cc_library(
    name = "trace_events_util",
    hdrs = ["trace_events_util.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events_util",
    ],
)

cc_library(
    name = "trace_events",
    hdrs = ["trace_events.h"],
    deps = [
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
    ],
)
