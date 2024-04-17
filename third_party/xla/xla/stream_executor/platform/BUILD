load("@local_tsl//tsl/platform:build_config.bzl", "tf_stream_executor_deps")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "platform",
    textual_hdrs = [
        "initialize.h",
        "platform.h",
        "port.h",
    ],
    deps = [
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ] + tf_stream_executor_deps("platform", "//xla/stream_executor/platform/"),
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ] + tf_stream_executor_deps("dso_loader", "//xla/stream_executor/platform/"),
)
