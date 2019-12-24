load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

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

tf_proto_library(
    name = "xplane_proto",
    srcs = ["xplane.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = [":friends"],
)

# This is needed because of how tf_android_core_proto_sources parses proto paths.
exports_files(
    srcs = ["xplane.proto"],
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_proto_library(
    name = "input_pipeline_proto",
    srcs = ["input_pipeline.proto"],
    cc_api_version = 2,
    protodeps = [":hardware_types_proto"],
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
    name = "steps_db_proto",
    srcs = ["steps_db.proto"],
    cc_api_version = 2,
    protodeps = [":op_metrics_proto"],
    visibility = [
        ":friends",
    ],
)

tf_proto_library(
    name = "op_stats_proto",
    srcs = ["op_stats.proto"],
    cc_api_version = 2,
    protodeps = [
        ":op_metrics_proto",
        ":steps_db_proto",
    ],
    visibility = [
        ":friends",
    ],
)

# This proto is deprecating and not guarenteed to be compatible across versions.
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
