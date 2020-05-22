# Experimental cache filesystem helper.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ram_blockcache",
    srcs = ["ram_blockcache.cc"],
    hdrs = ["ram_blockcache.h"],
    deps = [
        ":blockcache",
        ":cleanup",
        ":notification",
        ":thread_annotations",
        "//tensorflow/c:tf_status",
    ],
)

cc_library(
    name = "blockcache",
    hdrs = ["blockcache.h"],
    deps = [
        "//tensorflow/c:tf_status",
    ],
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
)

tf_cc_test(
    name = "ram_blockcache_test",
    size = "small",
    srcs = ["ram_blockcache_test.cc"],
    deps = [
        ":ram_blockcache",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform/cloud:now_seconds_env",
    ],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    deps = [
        ":thread_annotations",
        "//tensorflow/c:tf_status",
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
