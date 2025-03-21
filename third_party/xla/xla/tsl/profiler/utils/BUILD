load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla/tsl/profiler:friends",
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
        "//xla/tsl/platform:logging",
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
        "//xla:__subpackages__",
        "//xla/tsl/platform/cloud:__pkg__",
        "//xla/tsl/profiler:internal",
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
    visibility = internal_visibility([":friends"]),
    deps = [
        ":math_utils",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//xla/tsl/platform:macros",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:regexp",
    ],
)

tsl_cc_test(
    name = "tf_op_utils_test",
    size = "small",
    srcs = ["tf_op_utils_test.cc"],
    deps = [
        ":tf_op_utils",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "trace_filter_utils",
    hdrs = ["trace_filter_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
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
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        ":xplane_schema",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "xplane_builder_test",
    size = "small",
    srcs = ["xplane_builder_test.cc"],
    deps = [
        ":xplane_builder",
        ":xplane_visitor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/backends/profiler/gpu:__pkg__",
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "//xla/tsl/platform:types",
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
        "//xla/tsl/platform:types",
        "//xla/tsl/util:stats_calculator_portable",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/profiler/lib:context_types",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "xplane_utils_test",
    srcs = ["xplane_utils_test.cc"],
    deps = [
        ":math_utils",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        ":timespan",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "tpu_xplane_utils",
    srcs = ["tpu_xplane_utils.cc"],
    hdrs = ["tpu_xplane_utils.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xplane_schema",
        ":xplane_utils",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "tpu_xplane_utils_test",
    srcs = ["tpu_xplane_utils_test.cc"],
    deps = [
        ":tpu_xplane_utils",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "file_system_utils",
    hdrs = ["file_system_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "buffer_pool",
    srcs = ["buffer_pool.cc"],
    hdrs = ["buffer_pool.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/backends/profiler/gpu:__pkg__",
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "//xla/tsl/platform:logging",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tsl_cc_test(
    name = "buffer_pool_test",
    srcs = ["buffer_pool_test.cc"],
    deps = [
        ":buffer_pool",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
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
        ":xplane_mutators",
        ":xplane_schema",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:context_types",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "session_manager",
    srcs = ["session_manager.cc"],
    hdrs = ["session_manager.h"],
    deps = [
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "@com_google_absl//absl/log",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "no_init.h",
        "trace_filter_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "no_init",
    hdrs = ["no_init.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "lock_free_queue",
    hdrs = ["lock_free_queue.h"],
    deps = [
        ":no_init",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
    ],
)

tsl_cc_test(
    name = "lock_free_queue_test",
    size = "small",
    timeout = "short",
    srcs = ["lock_free_queue_test.cc"],
    deps = [
        ":lock_free_queue",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "per_thread",
    hdrs = ["per_thread.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "xplane_mutators",
    hdrs = ["xplane_mutators.h"],
    deps = [":xplane_builder"],
)

tsl_cc_test(
    name = "per_thread_test",
    size = "small",
    timeout = "short",
    srcs = ["per_thread_test.cc"],
    deps = [
        ":per_thread",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "device_utils",
    srcs = ["device_utils.cc"],
    hdrs = ["device_utils.h"],
    deps = [
        ":xplane_schema",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "device_utils_test",
    srcs = ["device_utils_test.cc"],
    deps = [
        ":device_utils",
        ":xplane_schema",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "profiler_options_util",
    srcs = ["profiler_options_util.cc"],
    hdrs = ["profiler_options_util.h"],
    deps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
)

tsl_cc_test(
    name = "profiler_options_util_test",
    srcs = ["profiler_options_util_test.cc"],
    deps = [
        ":profiler_options_util",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
)
