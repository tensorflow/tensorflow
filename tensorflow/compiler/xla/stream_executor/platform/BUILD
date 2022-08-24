load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_stream_executor_deps")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = [":friends"],
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
        "logging.h",
        "platform.h",
        "port.h",
    ],
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/platform:logging",
    ] + tf_stream_executor_deps("platform", "//tensorflow/compiler/xla/stream_executor/platform/"),
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ] + tf_stream_executor_deps("dso_loader", "//tensorflow/compiler/xla/stream_executor/platform/"),
)
