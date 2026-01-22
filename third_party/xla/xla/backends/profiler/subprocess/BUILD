load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_binary", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "subprocess_registry",
    srcs = ["subprocess_registry.cc"],
    hdrs = ["subprocess_registry.h"],
    visibility = internal_visibility([
        "//xla/python:__pkg__",
    ]),
    deps = [
        "//xla/tsl/platform:env",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
    ],
)

xla_cc_test(
    name = "subprocess_registry_test",
    srcs = ["subprocess_registry_test.cc"],
    deps = [
        ":subprocess_registry",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "subprocess_profiling_session",
    srcs = ["subprocess_profiling_session.cc"],
    hdrs = ["subprocess_profiling_session.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
    ]),
    deps = [
        ":subprocess_registry",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/utils:timestamp_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/profiler/lib:profiler_collection",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "subprocess_profiling_session_test",
    srcs = ["subprocess_profiling_session_test.cc"],
    args = ["--vmodule=subprocess_profiling_session=10"],
    data = [":subprocess_main"],
    env = {
        "XPROF_TEST_SUBPROCESS_MAIN_PATH": "$(rlocationpath :subprocess_main)",
    },
    deps = [
        ":subprocess_profiling_session",
        ":subprocess_registry",
        "//xla/backends/profiler:profiler_backends",  # buildcleaner: keep
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/profiler/utils:math_utils",
        "//xla/tsl/profiler/utils:timespan",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_binary(
    name = "subprocess_main",
    testonly = True,
    srcs = ["subprocess_main.cc"],
    visibility = internal_visibility([
        "//xla/backends/profiler/subprocess:__pkg__",
    ]),
    deps = [
        "//xla/backends/profiler:profiler_backends",  # buildcleaner: keep
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)
