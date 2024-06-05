load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add are: all tests.

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:types",
    ],
)

cc_library(
    name = "timed",
    hdrs = [
        "timed.h",
    ],
    deps = [
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:timed",
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
        "@local_tsl//tsl/lib/monitoring:cell_reader",
    ],
)

cc_library(
    name = "cell_reader-inl",
    testonly = 1,
    hdrs = ["cell_reader-inl.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":collected_metrics",
        ":collection_registry",
        ":metric_def",
        ":test_utils",
        ":types",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/monitoring:cell_reader-inl",
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
        "//tensorflow/core/framework:summary_proto_cc",
        "@local_tsl//tsl/lib/monitoring:collected_metrics",
    ],
)

cc_library(
    name = "collection_registry",
    hdrs = ["collection_registry.h"],
    visibility = [
        "//learning/brain/google/monitoring:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        ":collected_metrics",
        ":metric_def",
        ":types",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:collection_registry",
    ],
)

cc_library(
    name = "counter",
    hdrs = ["counter.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:counter",
    ],
)

cc_library(
    name = "gauge",
    hdrs = ["gauge.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/core/lib/core:status",
        "@local_tsl//tsl/lib/monitoring:gauge",
    ],
)

cc_library(
    name = "metric_def",
    hdrs = ["metric_def.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":types",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:metric_def",
    ],
)

cc_library(
    name = "sampler",
    hdrs = ["sampler.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/histogram",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/lib/monitoring:sampler",
    ],
)

cc_library(
    name = "percentile_sampler",
    hdrs = ["percentile_sampler.h"],
    deps = [
        ":collection_registry",
        ":metric_def",
        ":types",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "@local_tsl//tsl/lib/monitoring:percentile_sampler",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    hdrs = ["test_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tensorflow/core/framework:summary_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/monitoring:test_utils",
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
        "test_utils.h",
        "timed.h",
        "types.h",
        "@local_tsl//tsl/lib/monitoring:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
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
        "@local_tsl//tsl/lib/monitoring:legacy_lib_monitoring_lib_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
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
        "@local_tsl//tsl/lib/monitoring:legacy_lib_monitoring_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# Note(bmzhao): Ideally we would use a filegroup to represent these tests instead.
# However, that causes tf_cc_tests to link all of these tests into a single object
# file. This breaks collection_registry_test, because sample_test.cc has static variables
# that instantiate metrics with the same names that collection_registry_test tries
# to create ("/tensorflow/test/sampler_with_labels" and
# "/tensorflow/test/sampler_without_labels").
exports_files(
    [
        "cell_reader_test.cc",
        "collection_registry_test.cc",
        "counter_test.cc",
        "gauge_test.cc",
        "metric_def_test.cc",
        "percentile_sampler_test.cc",
        "sampler_test.cc",
        "test_utils_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
