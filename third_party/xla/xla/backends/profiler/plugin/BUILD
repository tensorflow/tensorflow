load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
        "//xla/tsl/platform:logging",
        "@tsl//tsl/profiler/lib:profiler_collection",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_factory_impl",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:logging",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
