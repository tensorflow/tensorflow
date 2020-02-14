load("//tensorflow/core/platform:build_config.bzl", "tf_platform_deps")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
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
        "thread_annotations.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
    ] + tf_platform_deps("platform", "//tensorflow/stream_executor/platform/"),
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("dso_loader", "//tensorflow/stream_executor/platform/"),
)
