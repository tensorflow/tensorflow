load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//learning/brain/google/monitoring:__pkg__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/platform:monitoring depends on this package
        "//tensorflow/core/platform:__subpackages__",
        # tensorflow/compiler/xla/pjrt:metrics depends on this package
        "//tensorflow/compiler/xla/pjrt:__subpackages__",
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        # tensorflow/compiler/mlir/tfrt:tf_jitrt depends on this package
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/lib/monitoring:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "counter",
    hdrs = ["counter.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "gauge",
    hdrs = [
        "gauge.h",
    ],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/tsl/lib/histogram",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "metric_def",
    hdrs = ["metric_def.h"],
    visibility = [
        "//learning/brain/google/monitoring:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        ":types",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "collection_registry",
    srcs = ["collection_registry.cc"],
    hdrs = ["collection_registry.h"],
    visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        ":metric_def",
        ":types",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/lib/monitoring:collected_metrics",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "sampler.h",
        "types.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_monitoring_lib_headers",
    srcs = [
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "sampler.h",
        "types.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_monitoring_all_headers",
    srcs = [
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "sampler.h",
        "types.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ],
)
