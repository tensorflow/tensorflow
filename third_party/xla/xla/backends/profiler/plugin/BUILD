load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)

package(
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
    visibility = ["//xla:internal"],
    deps = [
        ":profiler_c_api_hdrs",
        "//xla:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_error",
    srcs = ["profiler_error.cc"],
    hdrs = ["profiler_error.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":profiler_c_api_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "plugin_tracer_impl",
    srcs = ["plugin_tracer_impl.cc"],
    hdrs = ["plugin_tracer_impl.h"],
    visibility = [
        "//learning/brain/research/pjrt:__pkg__",
        "//xla/pjrt/c:__pkg__",
    ],
    deps = [
        ":profiler_c_api_hdrs",
        ":profiler_error",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:profiler_collection",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "plugin_tracer_impl_test",
    srcs = ["plugin_tracer_impl_test.cc"],
    deps = [
        ":plugin_tracer",
        ":plugin_tracer_impl",
        ":profiler_c_api_hdrs",
        ":profiler_error",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_visitor",
    ],
)
