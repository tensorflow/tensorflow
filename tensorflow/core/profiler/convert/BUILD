load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

# DO NOT ADD NEW DEPENDENCIES TO ANY TARGET IN THIS FILE.
# Instead, use //third_party/xprof/convert.

cc_library(
    name = "hlo_proto_to_memory_visualization_utils",
    hdrs = ["hlo_proto_to_memory_visualization_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "@org_xprof//xprof/convert:hlo_proto_to_memory_visualization_utils",
    ],
)

cc_library(
    name = "profile_time_breakdown",
    hdrs = ["profile_time_breakdown.h"],
    copts = tf_profiler_copts(),
    deps = [
        "@org_xprof//xprof/convert:profile_time_breakdown",
    ],
)

cc_library(
    name = "xplane_to_op_stats",
    hdrs = ["xplane_to_op_stats.h"],
    copts = tf_profiler_copts(),
    deps = [
        "@org_xprof//xprof/convert:xplane_to_op_stats",
    ],
)
