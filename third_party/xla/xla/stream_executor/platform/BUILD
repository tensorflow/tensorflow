load("@local_tsl//tsl:tsl.bzl", "set_external_visibility")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_stream_executor_deps")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ] + tf_stream_executor_deps("platform", "//xla/stream_executor/platform/"),
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
    ] + tf_stream_executor_deps("dso_loader", "//xla/stream_executor/platform/"),
)
