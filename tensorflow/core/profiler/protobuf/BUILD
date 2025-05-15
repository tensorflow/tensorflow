# Placeholder: load py_proto_library
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

package_group(
    name = "memory_viewer_friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
    packages = [
        "//learning/deepmind/jax/statix/...",
        "//platforms/xla/tools/...",
        "//smartass/brain/tpu_worker/...",
    ],
)

# This is needed because of how tf_android_core_proto_sources parses proto paths.
exports_files(
    srcs = ["xplane.proto"],
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_proto_library(
    name = "diagnostics_proto",
    srcs = ["diagnostics.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:diagnostics_proto"],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:diagnostics_proto",
    ],
)

tf_proto_library(
    name = "topology_proto",
    srcs = ["topology.proto"],
    protodeps = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:topology_proto",
    ],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:topology_proto",
    ],
)

tf_proto_library(
    name = "input_pipeline_proto",
    srcs = ["input_pipeline.proto"],
    protodeps = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:input_pipeline_proto",
    ],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:input_pipeline_proto",
    ],
)

tf_proto_library(
    name = "overview_page_proto",
    srcs = ["overview_page.proto"],
    protodeps = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:overview_page_proto",
    ],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:overview_page_proto",
    ],
)

tf_proto_library(
    name = "op_metrics_proto",
    srcs = ["op_metrics.proto"],
    protodeps = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto",
    ],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto",
    ],
)

tf_proto_library(
    name = "power_metrics_proto",
    srcs = ["power_metrics.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:power_metrics_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:power_metrics_proto",
    ],
)

tf_proto_library(
    name = "pod_stats_proto",
    srcs = ["pod_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:pod_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:pod_stats_proto",
    ],
)

tf_proto_library(
    name = "pod_viewer_proto",
    srcs = ["pod_viewer.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:pod_viewer_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:pod_viewer_proto",
    ],
)

tf_proto_library(
    name = "steps_db_proto",
    srcs = ["steps_db.proto"],
    protodeps = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto",
    ],
    visibility = [
        ":friends",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto",
    ],
)

tf_proto_library(
    name = "op_profile_proto",
    srcs = ["op_profile.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_profile_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_profile_proto",
    ],
)

tf_proto_library(
    name = "hlo_stats_proto",
    srcs = ["hlo_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hlo_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hlo_stats_proto",
    ],
)

tf_proto_library(
    name = "roofline_model_proto",
    srcs = ["roofline_model.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:roofline_model_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:roofline_model_proto",
    ],
)

tf_proto_library(
    name = "op_stats_proto",
    srcs = ["op_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_stats_proto",
    ],
)

tf_proto_library(
    name = "kernel_stats_proto",
    srcs = ["kernel_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:kernel_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:kernel_stats_proto",
    ],
)

tf_proto_library(
    name = "tf_function_proto",
    srcs = ["tf_function.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_function_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_function_proto",
    ],
)

# This proto is deprecating and not guaranteed to be compatible across versions.
# Please don't refer in new project unless you are double confirmed.
tf_proto_library(
    name = "tf_stats_proto",
    srcs = ["tf_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_stats_proto",
    ],
)

tf_proto_library(
    name = "hardware_types_proto",
    srcs = ["hardware_types.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hardware_types_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hardware_types_proto",
    ],
)

tf_proto_library(
    name = "tfstreamz_proto",
    srcs = ["tfstreamz.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tfstreamz_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tfstreamz_proto",
    ],
)

tf_proto_library(
    name = "memory_profile_proto",
    srcs = ["memory_profile.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:memory_profile_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:memory_profile_proto",
    ],
)

tf_proto_library(
    name = "tf_data_stats_proto",
    srcs = ["tf_data_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_data_stats_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_data_stats_proto",
    ],
)

tf_proto_library(
    name = "memory_viewer_preprocess_proto",
    srcs = ["memory_viewer_preprocess.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:memory_viewer_preprocess_proto"],
    visibility = [":memory_viewer_friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:memory_viewer_preprocess_proto",
    ],
)

# copybara:uncomment_begin(google-only)
#
# py_proto_library(
#     name = "memory_viewer_preprocess_py_pb2",
#     visibility = [":memory_viewer_friends"],
#     deps = [":memory_viewer_preprocess_proto"],
# )
#
# py_proto_library(
#     name = "op_profile_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":op_profile_proto"],
# )
#
# py_proto_library(
#     name = "hlo_stats_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":hlo_stats_proto"],
# )
#
# py_proto_library(
#     name = "roofline_model_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":roofline_model_proto"],
# )
#
# py_proto_library(
#     name = "op_metrics_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":op_metrics_proto"],
# )
#
# copybara:uncomment_end
tf_proto_library(
    name = "task_proto",
    srcs = ["task.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:task_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:task_proto",
    ],
)

tf_proto_library(
    name = "trace_events_proto",
    srcs = ["trace_events.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_proto",
    ],
)

tf_proto_library(
    name = "trace_events_raw_proto",
    srcs = ["trace_events_raw.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_raw_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_raw_proto",
    ],
)

tf_proto_library(
    name = "dcn_slack_analysis_proto",
    srcs = ["dcn_slack_analysis.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_slack_analysis_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_slack_analysis_proto",
    ],
)

tf_proto_library(
    name = "dcn_collective_info_proto",
    srcs = ["dcn_collective_info.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_collective_info_proto"],
    visibility = [":friends"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_collective_info_proto",
    ],
)

tf_proto_library(
    name = "inference_stats_proto",
    srcs = ["inference_stats.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:inference_stats_proto"],
    visibility = [
        ":friends",
        "//learning/serving/tools/servo_model_profiler:__subpackages__",
    ],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:inference_stats_proto",
    ],
)

tf_proto_library(
    name = "tpu_input_pipeline_proto",
    srcs = ["tpu_input_pipeline.proto"],
    protodeps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tpu_input_pipeline_proto"],
    exports = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tpu_input_pipeline_proto",
    ],
)
