package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "run_metadata_to_trace_events",
    srcs = ["run_metadata_to_trace_events.cc"],
    hdrs = ["run_metadata_to_trace_events.h"],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_metrics_to_record",
    srcs = ["op_metrics_to_record.cc"],
    hdrs = ["op_metrics_to_record.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "op_stack",
    hdrs = ["op_stack.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "op_stats_to_tf_stats",
    srcs = ["op_stats_to_tf_stats.cc"],
    hdrs = ["op_stats_to_tf_stats.h"],
    deps = [
        "//tensorflow/core/profiler/convert:op_metrics_to_record",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)
