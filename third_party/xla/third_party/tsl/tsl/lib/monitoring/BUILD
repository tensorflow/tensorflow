load("//tsl:tsl.bzl", "set_external_visibility")
load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "counter",
    hdrs = ["counter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tsl/platform",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "gauge",
    hdrs = [
        "gauge.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tsl/platform",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tsl/lib/histogram",
        "//tsl/platform",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "metric_def",
    hdrs = ["metric_def.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
        "//tsl/protobuf:histogram_proto_cc",
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
        "//tsl/platform",
        "//tsl/platform:env",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:stringpiece",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "collected_metrics",
    hdrs = [
        "collected_metrics.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metric_def",
        ":types",
        "//tsl/protobuf:histogram_proto_cc",
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
    visibility = ["//visibility:public"],
    #visibility = ["//visibility:private"],
    deps = [
        ":collected_metrics",
        ":collection_registry",
        ":metric_def",
        ":test_utils",
        ":types",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "percentile_sampler",
    srcs = ["percentile_sampler.cc"],
    hdrs = ["percentile_sampler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_registry",
        ":metric_def",
        ":types",
        "//tsl/platform",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
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
        "//tsl/platform:errors",
        "//tsl/platform:statusor",
        "//tsl/protobuf:histogram_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "timed",
    hdrs = [
        "timed.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:env_time",
        "//tsl/platform:types",
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
