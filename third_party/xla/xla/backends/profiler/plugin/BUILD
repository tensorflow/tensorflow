load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "profiler_c_api_hdrs",
    hdrs = ["profiler_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "plugin_tracer",
    srcs = ["plugin_tracer.cc"],
    hdrs = ["plugin_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":profiler_c_api_hdrs",
        "//xla:status",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)
