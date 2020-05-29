load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "event_span",
    srcs = ["event_span.cc"],
    hdrs = ["event_span.h"],
    deps = [
        ":timespan",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hardware_type_utils",
    srcs = ["hardware_type_utils.cc"],
    hdrs = ["hardware_type_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
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
    deps = [
        ":math_utils",
        ":tf_op_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_utils",
    srcs = ["op_utils.cc"],
    hdrs = ["op_utils.h"],
    deps = [
        ":op_metrics_db_utils",
        ":tf_op_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    deps = [
        "//tensorflow/core:regexp_internal",
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
    deps = [
        ":time_utils",
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
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    visibility = [":friends"],
    deps = [
        ":time_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
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
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    visibility = [":friends"],
    deps = [
        ":timespan",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core:platform_base",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xplane_utils_test",
    srcs = ["xplane_utils_test.cc"],
    deps = [
        ":xplane_builder",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:platform_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
    visibility = [":friends"],
    deps = [
        ":time_utils",
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
    visibility = [":friends"],
    deps = [
        ":tf_op_utils",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
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
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "cost_utils",
    srcs = ["cost_utils.cc"],
    hdrs = ["cost_utils.h"],
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
    deps = [
        ":group_events",
        ":tf_op_utils",
        ":tf_xplane_visitor",
        ":time_utils",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
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
        ":xplane_utils",
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
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tfstreamz_utils",
    srcs = ["tfstreamz_utils.cc"],
    hdrs = ["tfstreamz_utils.h"],
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
