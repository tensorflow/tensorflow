load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
        "//learning/brain/google/monitoring:__subpackages__",
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
        "//tensorflow/compiler/xla/hlo/experimental:__subpackages__",
        "//tensorflow/core/lib/monitoring:__subpackages__",
        "//tensorflow/compiler/xla/service:__subpackages__",
        "//tensorflow/tsl/framework:__subpackages__",
        "//tensorflow/tsl/distributed_runtime:__subpackages__",
    ]),
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
        "//tensorflow/tsl/lib/histogram",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
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
    visibility = set_external_visibility([
        "//learning/brain/google/monitoring:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ]),
    deps = [
        ":types",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "collection_registry",
    srcs = ["collection_registry.cc"],
    hdrs = ["collection_registry.h"],
    visibility = set_external_visibility([
        "//tensorflow/core:__subpackages__",
    ]),
    deps = [
        ":collected_metrics",
        ":metric_def",
        ":types",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "collected_metrics",
    hdrs = [
        "collected_metrics.h",
    ],
    deps = [
        ":metric_def",
        ":types",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "cell_reader",
    testonly = 1,
    hdrs = ["cell_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cell_reader-inl",
        ":collected_metrics",
        ":metric_def",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "cell_reader-inl",
    testonly = 1,
    srcs = ["cell_reader-inl.cc"],
    hdrs = ["cell_reader-inl.h"],
    #visibility = ["//visibility:private"],
    deps = [
        ":collected_metrics",
        ":collection_registry",
        ":metric_def",
        ":types",
        "//tensorflow/tsl/lib/monitoring:test_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "percentile_sampler",
    srcs = ["percentile_sampler.cc"],
    hdrs = ["percentile_sampler.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        ":types",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:histogram_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "timed",
    hdrs = [
        "timed.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:env_time",
        "//tensorflow/tsl/platform:types",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "cell_reader.h",
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "percentile_sampler.h",
        "sampler.h",
        "timed.h",
        "types.h",
    ],
    visibility = set_external_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ]),
)

filegroup(
    name = "legacy_lib_monitoring_lib_headers",
    srcs = [
        "cell_reader.h",
        "collected_metrics.h",
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "percentile_sampler.h",
        "sampler.h",
        "test_utils.h",
        "timed.h",
        "types.h",
    ],
    visibility = set_external_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ]),
)

filegroup(
    name = "legacy_lib_monitoring_all_headers",
    srcs = [
        "cell_reader.h",
        "collected_metrics.h",
        "collection_registry.h",
        "counter.h",
        "gauge.h",
        "metric_def.h",
        "percentile_sampler.h",
        "sampler.h",
        "test_utils.h",
        "types.h",
    ],
    visibility = set_external_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ]),
)
