load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/tsl/profiler:internal",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/tsl/profiler:friends",
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
        "//tensorflow/tsl/platform:logging",
    ],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
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
    visibility = [
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/tsl/platform/cloud:__pkg__",
        "//tensorflow/tsl/profiler:internal",
    ],
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
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "tf_op_utils_test",
    size = "small",
    srcs = ["tf_op_utils_test.cc"],
    deps = [
        ":tf_op_utils",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_schema",
    srcs = ["xplane_schema.cc"],
    hdrs = ["xplane_schema.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":tf_op_utils",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:context_types_hdrs",
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
    visibility = [":friends"],
    deps = [
        ":timespan",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
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
    visibility = [":friends"],
    deps = [
        ":math_utils",
        ":timespan",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler/gpu:__pkg__",
        "//tensorflow/tsl/profiler:internal",
    ],
    deps = [
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":math_utils",
        ":tf_xplane_visitor",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/tsl/platform:fingerprint",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:context_types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/util:stats_calculator_portable",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:tf_xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_xplane_visitor",
    hdrs = ["tf_xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "parse_annotation",
    srcs = ["parse_annotation.cc"],
    hdrs = ["parse_annotation.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "parse_annotation_test",
    srcs = ["parse_annotation_test.cc"],
    deps = [
        ":parse_annotation",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "group_events",
    srcs = ["group_events.cc"],
    hdrs = ["group_events.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
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
    visibility = [":friends"],
    deps = [
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
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
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
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
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "tpu_xplane_utils_test",
    srcs = ["tpu_xplane_utils_test.cc"],
    deps = [
        ":tpu_xplane_utils",
        ":xplane_schema",
        ":xplane_utils",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "file_system_utils",
    hdrs = ["file_system_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/tsl/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "buffer_pool",
    srcs = ["buffer_pool.cc"],
    hdrs = ["buffer_pool.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler/gpu:__pkg__",
        "//tensorflow/tsl/profiler:internal",
    ],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:thread_annotations",
    ],
)

tsl_cc_test(
    name = "buffer_pool_test",
    srcs = ["buffer_pool_test.cc"],
    deps = [
        ":buffer_pool",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)
