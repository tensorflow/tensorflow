load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/platform:build_config_root.bzl", "if_static")
load("//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "traceme_recorder",
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
    ] + if_static([
        ":traceme_recorder_impl",
    ]),
)

cc_library(
    name = "traceme_recorder_impl",
    srcs = [
        "traceme_recorder.cc",
    ],
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "//tsl/platform/cloud:__pkg__",
        "//tsl/profiler:__pkg__",
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_internal",
    ]),
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/profiler/utils:lock_free_queue",
        "//tsl/profiler/utils:per_thread",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "traceme_recorder_test",
    srcs = ["traceme_recorder_test.cc"],
    deps = [
        ":traceme_recorder",
        ":traceme_recorder_impl",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:logging",
        "//tsl/platform:notification",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
        "//tsl/profiler/utils:math_utils",
        "//tsl/profiler/utils:time_utils",
        "//tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "annotation_stack",
    hdrs = ["annotation_stack.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ] + if_static([
        ":annotation_stack_impl",
    ]),
)

cc_library(
    name = "annotation_stack_impl",
    srcs = [
        "annotation_stack.cc",
        "annotation_stack.h",
    ],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla:__subpackages__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_utils",
    srcs = ["host_tracer_utils.cc"],
    hdrs = ["host_tracer_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_internal",
    ]),
    deps = [
        ":traceme_recorder",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:parse_annotation",
        "//tsl/profiler/utils:tf_op_utils",
        "//tsl/profiler/utils:xplane_builder",
        "//tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "threadpool_listener",
    srcs = ["threadpool_listener.cc"],
    hdrs = ["threadpool_listener.h"],
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "//tsl/platform/cloud:__pkg__",
        "//tsl/profiler:__pkg__",
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_internal",
    ]),
    deps = [
        ":threadpool_listener_state",
        ":traceme_recorder",
        "//tsl/platform:logging",
        "//tsl/platform:tracing",
        "//tsl/platform:types",
        "//tsl/profiler/lib:context_types_hdrs",
        "//tsl/profiler/lib:profiler_interface",
        "//tsl/profiler/lib:traceme_encode",
        "//tsl/profiler/utils:time_utils",
        "//tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "threadpool_listener_state",
    srcs = ["threadpool_listener_state.cc"],
    hdrs = ["threadpool_listener_state.h"],
    visibility = internal_visibility([
        "//tsl/platform:__subpackages__",
    ]),
)
