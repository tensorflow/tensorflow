load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer_factory.cc"],
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:__pkg__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        ":host_tracer_impl",
        "//tensorflow/tsl/profiler/lib:profiler_factory",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_impl",
    srcs = ["host_tracer.cc"],
    hdrs = ["host_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/backends/cpu:host_tracer_utils",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder",
        "//tensorflow/tsl/profiler/lib:profiler_interface",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:time_utils",
        "//tensorflow/tsl/profiler/utils:xplane_schema",
        "//tensorflow/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "python_tracer",
    srcs = ["python_tracer_factory.cc"],
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        ":python_tracer_impl",
        "//tensorflow/tsl/profiler/lib:profiler_factory",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer_impl",
    srcs = ["python_tracer.cc"],
    hdrs = ["python_tracer.h"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/compiler/xla/python/profiler/internal:python_hooks",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/lib:profiler_interface",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:__pkg__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        ":metadata_utils",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/lib:profiler_factory",
        "//tensorflow/tsl/profiler/lib:profiler_interface",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:xplane_schema",
        "//tensorflow/tsl/profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_utils",
    hdrs = ["metadata_utils.h"],
    visibility = [
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/profiler/convert:xla_op_utils",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:xplane_builder",
        "//tensorflow/tsl/profiler/utils:xplane_schema",
    ],
)
