load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")

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
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/backends/cpu:host_tracer_utils",
        "@local_tsl//tsl/profiler/backends/cpu:threadpool_listener",
        "@local_tsl//tsl/profiler/backends/cpu:traceme_recorder",
        "@local_tsl//tsl/profiler/lib:profiler_collection",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:time_utils",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
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
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
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
        "@local_tsl//tsl/profiler/convert:xla_op_utils",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
    ],
)

xla_cc_test(
    name = "host_tracer_test",
    srcs = ["host_tracer_test.cc"],
    deps = [
        ":host_tracer_impl",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@local_tsl//tsl/profiler/utils:timespan",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_visitor",
    ],
)
