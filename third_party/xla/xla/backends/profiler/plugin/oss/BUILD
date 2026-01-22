load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "plugin_metadata_oss",
    srcs = ["plugin_metadata_oss.cc"],
    visibility = ["//xla/backends/profiler/plugin:__pkg__"],
    deps = [
        "//xla/backends/profiler/plugin:plugin_metadata_hdr",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)
