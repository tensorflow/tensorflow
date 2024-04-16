load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/platform:build_config_root.bzl", "if_static")
load("//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tsl/profiler:friends",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
)

cc_library(
    name = "format_utils",
    hdrs = ["format_utils.h"],
    deps = [
        "//tsl/platform:logging",
    ],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":math_utils",
    ] + if_static([
        ":time_utils_impl",
    ]),
)

cc_library(
    name = "time_utils_impl",
    srcs = [
        "time_utils.cc",
        "time_utils.h",
    ],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla:__subpackages__",
        "//tsl/platform/cloud:__pkg__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        ":math_utils",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":math_utils",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "tf_op_utils_test",
    size = "small",
    srcs = ["tf_op_utils_test.cc"],
    deps = [
        ":tf_op_utils",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_schema",
    srcs = ["xplane_schema.cc"],
    hdrs = ["xplane_schema.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":tf_op_utils",
        "//tsl/lib/gtl:map_util",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "//tsl/profiler/lib:context_types_hdrs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":timespan",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":math_utils",
        ":timespan",
        "//tsl/platform:macros",
        "//tsl/platform:protobuf",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tsl_cc_test(
    name = "xplane_builder_test",
    size = "small",
    srcs = ["xplane_builder_test.cc"],
    deps = [
        ":xplane_builder",
        ":xplane_visitor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla/backends/profiler/gpu:__pkg__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":math_utils",
        ":tf_xplane_visitor",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tsl/platform:fingerprint",
        "//tsl/platform:types",
        "//tsl/profiler/lib:context_types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

tsl_cc_test(
    name = "xplane_utils_test",
    srcs = ["xplane_utils_test.cc"],
    deps = [
        ":math_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:tf_xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_xplane_visitor",
    hdrs = ["tf_xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xplane_schema",
        ":xplane_visitor",
        "//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "parse_annotation",
    srcs = ["parse_annotation.cc"],
    hdrs = ["parse_annotation.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "parse_annotation_test",
    srcs = ["parse_annotation_test.cc"],
    deps = [
        ":parse_annotation",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "group_events",
    srcs = ["group_events.cc"],
    hdrs = ["group_events.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tsl/lib/gtl:map_util",
        "//tsl/platform:dso_loader",
        "//tsl/platform:env",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_test_utils",
    testonly = True,
    srcs = ["xplane_test_utils.cc"],
    hdrs = ["xplane_test_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

tsl_cc_test(
    name = "group_events_test",
    srcs = ["group_events_test.cc"],
    deps = [
        ":group_events",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_xplane_utils",
    srcs = ["tpu_xplane_utils.cc"],
    hdrs = ["tpu_xplane_utils.h"],
    deps = [
        ":xplane_schema",
        ":xplane_utils",
        "//tsl/platform:regexp",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tsl_cc_test(
    name = "tpu_xplane_utils_test",
    srcs = ["tpu_xplane_utils_test.cc"],
    deps = [
        ":tpu_xplane_utils",
        ":xplane_schema",
        ":xplane_utils",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_system_utils",
    hdrs = ["file_system_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla/python:__pkg__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "buffer_pool",
    srcs = ["buffer_pool.cc"],
    hdrs = ["buffer_pool.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla/backends/profiler/gpu:__pkg__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:mutex",
        "//tsl/platform:platform_port",
        "//tsl/platform:thread_annotations",
    ],
)

tsl_cc_test(
    name = "buffer_pool_test",
    srcs = ["buffer_pool_test.cc"],
    deps = [
        ":buffer_pool",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "preprocess_xplane",
    srcs = ["preprocess_xplane.cc"],
    hdrs = ["preprocess_xplane.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        ":tpu_xplane_utils",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        "//tsl/profiler/lib:context_types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tsl_cc_test(
    name = "preprocess_xplane_test",
    srcs = ["preprocess_xplane_test.cc"],
    deps = [
        ":preprocess_xplane",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/lib:connected_traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "session_manager",
    srcs = ["session_manager.cc"],
    hdrs = ["session_manager.h"],
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/profiler/lib:profiler_session",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "timestamp_utils",
    srcs = ["timestamp_utils.cc"],
    hdrs = ["timestamp_utils.h"],
    deps = [
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/log",
    ],
)

tsl_cc_test(
    name = "timestamp_utils_test",
    srcs = ["timestamp_utils_test.cc"],
    deps = [
        ":timestamp_utils",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)
