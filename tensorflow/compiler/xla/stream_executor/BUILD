# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

# TODO(rdzhabarov): XLA should have its own tensorflow independent rule.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//tensorflow/stream_executor:__subpackages__"],
    licenses = ["notice"],
)

# The stream_executor_headers target does not prescribe an implementation.
# TODO(rdzhabarov): Keep populating this based on the tensorflow/stream_executor:stream_executor_headers.
cc_library(
    name = "stream_executor_headers",
    textual_hdrs = [
        "device_description.h",
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/cuda:cuda_dnn_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "launch_dim",
    hdrs = [
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":launch_dim",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)
