load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_stream_executor_deps")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "initialize",
    hdrs = [
        "initialize.h",
    ],
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ] + tf_stream_executor_deps("platform", "//xla/stream_executor/platform/"),
)
