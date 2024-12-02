load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "traceme_recorder",
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
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
        "//xla/tsl/platform/cloud:__pkg__",
        "//xla/tsl/platform/cloud:__pkg__",
        "//xla/tsl/profiler:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:per_thread",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "traceme_recorder_test",
    srcs = ["traceme_recorder_test.cc"],
    deps = [
        ":traceme_recorder",
        ":traceme_recorder_impl",
        "//xla/tsl/profiler/utils:math_utils",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "annotation_stack",
    hdrs = ["annotation_stack.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
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
        "//xla:__subpackages__",
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_utils",
    srcs = ["host_tracer_utils.cc"],
    hdrs = ["host_tracer_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        ":traceme_recorder",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:tf_op_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "threadpool_listener",
    srcs = ["threadpool_listener.cc"],
    hdrs = ["threadpool_listener.h"],
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "//xla/tsl/platform/cloud:__pkg__",
        "//xla/tsl/platform/cloud:__pkg__",
        "//xla/tsl/profiler:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        ":threadpool_listener_state",
        ":traceme_recorder",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:tracing",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "threadpool_listener_state",
    srcs = ["threadpool_listener_state.cc"],
    hdrs = ["threadpool_listener_state.h"],
    visibility = internal_visibility([
        "@local_tsl//tsl/platform:__subpackages__",
        "//xla/tsl/platform:__subpackages__",
    ]),
)
