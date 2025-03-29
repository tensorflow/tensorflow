load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/google/data:__subpackages__",
        "//learning/brain/google/monitoring:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/experimental/orbax_model:__subpackages__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/platform:monitoring depends on this package
        "//tensorflow/core/platform:__subpackages__",
        # tensorflow/compiler/xla/pjrt:metrics depends on this package
        "//xla/pjrt:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        # tensorflow/compiler/mlir/tfrt:tf_jitrt depends on this package
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//xla/hlo/experimental:__subpackages__",
        "//tensorflow/core/lib/monitoring:__subpackages__",
        "//xla/service:__subpackages__",
        "//xla/tsl/framework:__subpackages__",
        "//xla/tsl/distributed_runtime:__subpackages__",
        "//tensorflow/compiler/mlir/tf2xla:__subpackages__",
        "//tensorflow_serving/model_servers:__subpackages__",
        # xprof depends on this package
        "//third_party/xprof:__subpackages__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "counter",
    hdrs = ["counter.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
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
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//xla/tsl/lib/histogram",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:histogram_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        "//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "metric_def",
    hdrs = ["metric_def.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:histogram_proto_cc",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "collection_registry",
    srcs = ["collection_registry.cc"],
    hdrs = ["collection_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collected_metrics",
        ":metric_def",
        ":types",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:histogram_proto_cc",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:thread_annotations",
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
        "//xla/tsl/protobuf:histogram_proto_cc",
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
        ":test_utils",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
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
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:histogram_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "timed",
    hdrs = [
        "timed.h",
    ],
    deps = [
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:types",
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
    visibility = internal_visibility([
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
    visibility = internal_visibility([
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
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/monitoring:__pkg__",
    ]),
)
