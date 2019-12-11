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
        "logging.h",
        "platform.h",
        "port.h",
        "thread_annotations.h",
        "initialize.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/platform/default:platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
        "//tensorflow/stream_executor/platform/default:dso_loader",
    ],
)
