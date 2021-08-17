load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental gcs filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object", "tf_cc_test")

package(
    licenses = ["notice"],
)

# Filesystem implementation for GCS environments
tf_cc_shared_object(
    name = "gcs_filesystem",
    framework_so = [],
    linkstatic = False,
    per_os_targets = 1,
    visibility = ["//visibility:public"],
    deps = [":gcs_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "gcs_filesystem_impl",
    srcs = ["gcs_filesystem.cc"],
    hdrs = ["gcs_filesystem.h"],
    copts = select({
        "//conditions:default": [],
        "//tensorflow:windows": get_win_copts(),
    }),
    deps = [
        ":expiring_lru_cache",
        ":gcs_helper",
        ":ram_file_block_cache",
        "//tensorflow/c:env",
        "//tensorflow/c:logging",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@com_github_googlecloudplatform_google_cloud_cpp//:storage_client",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "gcs_helper",
    srcs = ["gcs_helper.cc"],
    hdrs = ["gcs_helper.h"],
    linkstatic = 1,
    deps = [
        "//tensorflow/c:env",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
)

cc_library(
    name = "ram_file_block_cache",
    srcs = ["ram_file_block_cache.cc"],
    hdrs = ["ram_file_block_cache.h"],
    deps = [
        ":cleanup",
        "//tensorflow/c:env",
        "//tensorflow/c:logging",
        "//tensorflow/c:tf_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "ram_file_block_cache_test",
    size = "small",
    srcs = ["ram_file_block_cache_test.cc"],
    deps = [
        ":ram_file_block_cache",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform/cloud:now_seconds_env",
    ],
)

tf_cc_test(
    name = "gcs_filesystem_test",
    srcs = [
        "gcs_filesystem_test.cc",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":gcs_filesystem_impl",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    deps = [
        "//tensorflow/c:env",
        "//tensorflow/c:tf_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform/cloud:now_seconds_env",
    ],
)
