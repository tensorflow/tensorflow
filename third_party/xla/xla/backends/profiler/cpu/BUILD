load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer_factory.cc"],
    visibility = internal_visibility([
        "//xla/backends/profiler:__pkg__",
        # copybara:uncomment "//tensorflow/core/profiler:internal",
    ]),
    deps = [
        ":host_tracer_impl",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_impl",
    srcs = ["host_tracer.cc"],
    hdrs = ["host_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        # copybara:uncomment "//tensorflow/core/profiler:internal",
    ]),
    deps = [
        "//xla/tsl/profiler/backends/cpu:host_tracer_utils",
        "//xla/tsl/profiler/backends/cpu:threadpool_listener",
        "//xla/tsl/profiler/backends/cpu:traceme_recorder",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/profiler/lib:profiler_collection",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "python_tracer",
    srcs = ["python_tracer_factory.cc"],
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        # copybara:uncomment "//tensorflow/core/profiler:internal",
        # copybara:uncomment "//tensorflow:internal",
    ]),
    deps = [
        ":python_tracer_impl",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer_impl",
    srcs = ["python_tracer.cc"],
    hdrs = ["python_tracer.h"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([
        # copybara:uncomment "//tensorflow/core/profiler:internal",
    ]),
    deps = [
        "//xla/python/profiler/internal:python_hooks",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/backends/profiler:__pkg__",
        # copybara:uncomment "//tensorflow/core/profiler:internal",
    ]),
    deps = [
        ":metadata_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service:xla_debug_info_manager",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/status",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_utils",
    hdrs = ["metadata_utils.h"],
    visibility = internal_visibility([
        # copybara:uncomment "//tensorflow/core/profiler:internal",
    ]),
    deps = [
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/profiler/convert:xla_op_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "host_tracer_test",
    srcs = ["host_tracer_test.cc"],
    deps = [
        ":host_tracer_impl",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:timespan",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
