load("//tensorflow/core/platform:default/build_config.bzl", "tf_proto_library")

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
    visibility = [":friends"],
)

tf_proto_library(
    name = "op_metrics_proto",
    srcs = ["op_metrics.proto"],
    cc_api_version = 2,
    visibility = [":friends"],
)

tf_proto_library(
    name = "op_stats_proto",
    srcs = ["op_stats.proto"],
    cc_api_version = 2,
    protodeps = [
        ":op_metrics_proto",
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
