# tf.data service client library.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_profiler_service")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data/service:common_proto_cc",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "data_service_client",
    srcs = ["data_service_client.cc"],
    hdrs = ["data_service_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":validate_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/data/service:common",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/data/service:worker_client",
        "//tensorflow/core/data/service:worker_impl",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "data_service_client_test",
    srcs = ["data_service_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":data_service_client",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common",
        "//tensorflow/core/data/service:test_cluster",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/protobuf:protos_all_cc",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:test_cluster",
        "//tensorflow/core/data/service:test_util",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/protobuf:protos_all_cc",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "validate_utils",
    srcs = ["validate_utils.cc"],
    hdrs = ["validate_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data/service:common",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:worker_impl",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

tf_cc_test(
    name = "validate_utils_test",
    srcs = ["validate_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":validate_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:worker_impl",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
    ],
)
