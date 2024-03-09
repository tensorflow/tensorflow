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

tf_proto_library(
    name = "xplane_proto",
    srcs = ["xplane.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:xplane_proto",
    ],
    visibility = [":friends"],
    exports = [
        "@local_tsl//tsl/profiler/protobuf:xplane_proto",
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
    cc_api_version = 2,
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "topology_proto",
    srcs = ["topology.proto"],
    cc_api_version = 2,
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "input_pipeline_proto",
    srcs = ["input_pipeline.proto"],
    cc_api_version = 2,
    protodeps = [":diagnostics_proto"],
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "overview_page_proto",
    srcs = ["overview_page.proto"],
    cc_api_version = 2,
    protodeps = [
        ":diagnostics_proto",
        ":input_pipeline_proto",
        ":power_metrics_proto",
    ],
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "op_metrics_proto",
    srcs = ["op_metrics.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "power_metrics_proto",
    srcs = ["power_metrics.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "pod_stats_proto",
    srcs = ["pod_stats.proto"],
    cc_api_version = 2,
    protodeps = [
        ":diagnostics_proto",
    ],
    visibility = [":friends"],
)

tf_proto_library(
    name = "pod_viewer_proto",
    srcs = ["pod_viewer.proto"],
    cc_api_version = 2,
    protodeps = [
        ":diagnostics_proto",
        ":pod_stats_proto",
        ":topology_proto",
    ],
    visibility = [":friends"],
)

tf_proto_library(
    name = "steps_db_proto",
    srcs = ["steps_db.proto"],
    cc_api_version = 2,
    protodeps = [":op_metrics_proto"],
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "op_profile_proto",
    srcs = ["op_profile.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "op_stats_proto",
    srcs = ["op_stats.proto"],
    cc_api_version = 2,
    protodeps = [
        ":topology_proto",
        ":diagnostics_proto",
        ":kernel_stats_proto",
        ":op_metrics_proto",
        ":power_metrics_proto",
        ":steps_db_proto",
        ":tf_function_proto",
    ],
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "kernel_stats_proto",
    srcs = ["kernel_stats.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "tf_function_proto",
    srcs = ["tf_function.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

# This proto is deprecating and not guaranteed to be compatible across versions.
# Please don't refer in new project unless you are double confirmed.
tf_proto_library(
    name = "tf_stats_proto",
    srcs = ["tf_stats.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "hardware_types_proto",
    srcs = ["hardware_types.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "tfstreamz_proto",
    srcs = ["tfstreamz.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "memory_profile_proto",
    srcs = ["memory_profile.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "tf_data_stats_proto",
    srcs = ["tf_data_stats.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "memory_viewer_preprocess_proto",
    srcs = ["memory_viewer_preprocess.proto"],
    cc_api_version = 2,
    visibility = [":memory_viewer_friends"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xplane_py_pb2",
#     api_version = 2,
#     visibility = [":friends"],
#     deps = [":xplane_proto"],
# )
#
# py_proto_library(
#     name = "memory_viewer_preprocess_py_pb2",
#     api_version = 2,
#     visibility = [":memory_viewer_friends"],
#     deps = [":memory_viewer_preprocess_proto"],
# )
#
# py_proto_library(
#     name = "op_profile_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":op_profile_proto"],
# )
#
# py_proto_library(
#     name = "op_metrics_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":op_metrics_proto"],
# )
#
# copybara:uncomment_end
tf_proto_library(
    name = "task_proto",
    srcs = ["task.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "trace_events_proto",
    srcs = ["trace_events.proto"],
    cc_api_version = 2,
    protodeps = [
        ":task_proto",
    ],
    visibility = [":friends"],
)

tf_proto_library(
    name = "trace_events_raw_proto",
    srcs = ["trace_events_raw.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "dcn_slack_analysis_proto",
    srcs = ["dcn_slack_analysis.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "dcn_collective_info_proto",
    srcs = ["dcn_collective_info.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)
