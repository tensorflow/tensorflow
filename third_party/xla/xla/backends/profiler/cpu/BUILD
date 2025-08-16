load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
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
        "//xla/tsl/profiler/backends/cpu:host_tracer_utils",
        "//xla/tsl/profiler/backends/cpu:threadpool_listener",
        "//xla/tsl/profiler/backends/cpu:traceme_recorder",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/profiler/lib:profiler_collection",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
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
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

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
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/utils:file_system_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/profiler/lib:profiler_collection",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

xla_cc_test(
    name = "subprocess_profiling_session_test",
    srcs = ["subprocess_profiling_session_test.cc"],
    deps = [
        ":subprocess_profiling_session",
        ":subprocess_registry",
        "//xla/backends/profiler:profiler_backends",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/profiler/rpc:profiler_service_impl",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)
