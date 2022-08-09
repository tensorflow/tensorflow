# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

# TODO(rdzhabarov): XLA should have its own tensorflow independent rule.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    default_visibility = [
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/core/protobuf:__subpackages__",
        "//tensorflow/core/util:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
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
        "kernel_spec.h",
        "module_spec.h",
        "kernel_cache_config.h",
        "allocator_stats.h",
        "data_type.h",
        "device_memory.h",
        "device_options.h",
        "fft.h",
        "plugin.h",
        "rng.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/cuda:cuda_dnn_headers",
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
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":launch_dim",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "kernel_cache_config",
    hdrs = ["kernel_cache_config.h"],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "allocator_stats",
    srcs = [
        "allocator_stats.cc",
    ],
    hdrs = ["allocator_stats.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":dnn_proto_cc",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/platform"],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "plugin",
    srcs = ["plugin.cc"],
    hdrs = ["plugin.h"],
)

cc_library(
    name = "rng",
    srcs = ["rng.cc"],
    hdrs = ["rng.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)
