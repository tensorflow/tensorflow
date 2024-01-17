load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer_factory.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":host_tracer_impl",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_impl",
    srcs = ["host_tracer.cc"],
    hdrs = ["host_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/backends/cpu:host_tracer_utils",
        "@local_tsl//tsl/profiler/backends/cpu:traceme_recorder",
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
    visibility = ["//visibility:public"],
    deps = [
        ":python_tracer_impl",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla/python/profiler/internal:python_hooks",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    copts = tf_profiler_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service:xla_debug_info_manager",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_utils",
    hdrs = ["metadata_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@local_tsl//tsl/profiler/convert:xla_op_utils",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
    ],
)
