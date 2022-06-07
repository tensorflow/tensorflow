load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:diagnostics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "event_span",
    srcs = ["event_span.cc"],
    hdrs = ["event_span.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":timespan",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hardware_type_utils",
    srcs = ["hardware_type_utils.cc"],
    hdrs = ["hardware_type_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_schema",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "html_utils",
    hdrs = ["html_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_metrics_db_utils",
    srcs = ["op_metrics_db_utils.cc"],
    hdrs = ["op_metrics_db_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":math_utils",
        ":tf_op_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "op_metrics_db_utils_test",
    srcs = ["op_metrics_db_utils_test.cc"],
    deps = [
        ":op_metrics_db_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
    ],
)

cc_library(
    name = "op_utils",
    srcs = ["op_utils.cc"],
    hdrs = ["op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":op_metrics_db_utils",
        ":tf_op_utils",
        ":timespan",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/profiler/convert:op_metrics_db_combiner",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tf_op_utils_test",
    size = "small",
    srcs = ["tf_op_utils_test.cc"],
    deps = [
        ":tf_op_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":math_utils",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":math_utils",
        "//tensorflow/core:lib",
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
    deps = [
        ":math_utils",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_builder_test",
    size = "small",
    srcs = ["xplane_builder_test.cc"],
    deps = [
        ":xplane_builder",
        ":xplane_visitor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:context_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:context_types",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xplane_utils_test",
    srcs = ["xplane_utils_test.cc"],
    deps = [
        ":math_utils",
        ":xplane_builder",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
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
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "group_events_test",
    srcs = ["group_events_test.cc"],
    deps = [
        ":group_events",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "cost_utils",
    srcs = ["cost_utils.cc"],
    hdrs = ["cost_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":tf_op_utils",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:op_context",
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "//tensorflow/core/grappler/costs:op_performance_data_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "derived_timeline",
    srcs = ["derived_timeline.cc"],
    hdrs = ["derived_timeline.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":gpu_event_stats",
        ":group_events",
        ":tf_op_utils",
        ":tf_xplane_visitor",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/convert:xla_op_utils",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/util:stats_calculator_portable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "derived_timeline_test",
    srcs = ["derived_timeline_test.cc"],
    deps = [
        ":derived_timeline",
        ":group_events",
        ":tf_xplane_visitor",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_stats_utils",
    srcs = ["kernel_stats_utils.cc"],
    hdrs = ["kernel_stats_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "kernel_stats_utils_test",
    srcs = ["kernel_stats_utils_test.cc"],
    deps = [
        ":kernel_stats_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/backends/gpu:cupti_collector_header",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
    ],
)

cc_library(
    name = "tfstreamz_utils",
    srcs = ["tfstreamz_utils.cc"],
    hdrs = ["tfstreamz_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/profiler/protobuf:tfstreamz_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "step_intersection",
    srcs = ["step_intersection.cc"],
    hdrs = ["step_intersection.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":timespan",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "step_intersection_test",
    srcs = ["step_intersection_test.cc"],
    deps = [
        ":step_intersection",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "file_system_utils",
    hdrs = ["file_system_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "parse_annotation",
    srcs = ["parse_annotation.cc"],
    hdrs = ["parse_annotation.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:friends"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "parse_annotation_test",
    srcs = ["parse_annotation_test.cc"],
    deps = [
        ":parse_annotation",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_caps_utils",
    srcs = ["device_caps_utils.cc"],
    hdrs = ["device_caps_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "buffer_pool",
    srcs = ["buffer_pool.cc"],
    hdrs = ["buffer_pool.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "buffer_pool_test",
    srcs = ["buffer_pool_test.cc"],
    deps = [
        ":buffer_pool",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "gpu_event_stats",
    srcs = ["gpu_event_stats.cc"],
    hdrs = ["gpu_event_stats.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
