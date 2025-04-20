# Description:
# Cloud file system implementation.

load(
    "//xla/tsl:tsl.bzl",
    "if_windows",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":dependency_allowlist",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "dependency_allowlist",
    packages = [
        "//learning/brain/tfrc/...",
        "//tensorflow/...",
        "//third_party/gsmemcachedfs/...",
        "//third_party/py/tensorflow_io/...",
    ],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "file_block_cache",
    hdrs = ["file_block_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "ram_file_block_cache",
    srcs = ["ram_file_block_cache.cc"],
    hdrs = ["ram_file_block_cache.h"],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":file_block_cache",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "gcs_dns_cache",
    srcs = ["gcs_dns_cache.cc"],
    hdrs = ["gcs_dns_cache.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "gcs_throttle",
    srcs = ["gcs_throttle.cc"],
    hdrs = ["gcs_throttle.h"],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "gcs_file_system",
    srcs = ["gcs_file_system.cc"],
    hdrs = ["gcs_file_system.h"],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":compute_engine_metadata_client",
        ":compute_engine_zone_provider",
        ":curl_http_request",
        ":expiring_lru_cache",
        ":file_block_cache",
        ":gcs_dns_cache",
        ":gcs_throttle",
        ":google_auth_provider",
        ":http_request",
        ":ram_file_block_cache",
        ":time_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:file_statistics",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:retrying_file_system",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringprintf",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gcs_file_system_disabled",
    srcs = ["gcs_file_system.cc"],
    hdrs = ["gcs_file_system.h"],
    copts = tsl_copts() + [
        "-DTPU_GCS_FS",
        "-DPLATFORM_CLOUD_TPU",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":compute_engine_metadata_client",
        ":compute_engine_zone_provider",
        ":curl_http_request",
        ":expiring_lru_cache",
        ":file_block_cache",
        ":gcs_dns_cache",
        ":gcs_throttle",
        ":google_auth_provider",
        ":http_request",
        ":ram_file_block_cache",
        ":time_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:file_statistics",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:retrying_file_system",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringprintf",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    hdrs = ["http_request.h"],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "curl_http_request",
    srcs = ["curl_http_request.cc"],
    hdrs = ["curl_http_request.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "//xla/tsl/util:env_var",
        "@curl",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:scanner",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "http_request_fake",
    testonly = 1,
    hdrs = [
        "http_request_fake.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:types",
        "@curl",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "google_auth_provider",
    srcs = ["google_auth_provider.cc"],
    hdrs = [
        "auth_provider.h",
        "google_auth_provider.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":compute_engine_metadata_client",
        ":oauth_client",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:base64",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "compute_engine_metadata_client",
    srcs = [
        "compute_engine_metadata_client.cc",
    ],
    hdrs = [
        "compute_engine_metadata_client.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        ":http_request",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:retrying_utils",
    ],
)

cc_library(
    name = "compute_engine_zone_provider",
    srcs = [
        "compute_engine_zone_provider.cc",
    ],
    hdrs = [
        "compute_engine_zone_provider.h",
        "zone_provider.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":compute_engine_metadata_client",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@local_tsl//tsl/platform:str_util",
    ],
)

cc_library(
    name = "now_seconds_env",
    testonly = 1,
    hdrs = ["now_seconds_env.h"],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "oauth_client",
    srcs = [
        "oauth_client.cc",
    ],
    hdrs = [
        "oauth_client.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":curl_http_request",
        ":http_request",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@boringssl//:crypto",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:base64",
    ],
)

cc_library(
    name = "time_util",
    srcs = [
        "time_util.cc",
    ],
    hdrs = [
        "time_util.h",
    ],
    copts = tsl_copts(),
    deps = [
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        ":now_seconds_env",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "ram_file_block_cache_test",
    size = "small",
    srcs = ["ram_file_block_cache_test.cc"],
    deps = [
        ":now_seconds_env",
        ":ram_file_block_cache",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:notification",
    ],
)

tsl_cc_test(
    name = "gcs_file_system_test",
    size = "small",
    srcs = ["gcs_file_system_test.cc"],
    deps = [
        ":gcs_file_system",
        ":http_request_fake",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//xla/tsl/profiler/utils:time_utils_impl",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
    ],
)

tsl_cc_test(
    name = "gcs_dns_cache_test",
    size = "small",
    srcs = ["gcs_dns_cache_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_dns_cache",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:str_util",
    ],
)

tsl_cc_test(
    name = "gcs_throttle_test",
    size = "small",
    srcs = ["gcs_throttle_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_throttle",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:str_util",
    ],
)

tsl_cc_test(
    name = "curl_http_request_test",
    size = "small",
    srcs = ["curl_http_request_test.cc"],
    shuffle_tests = False,
    deps = [
        ":curl_http_request",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

tsl_cc_test(
    name = "oauth_client_test",
    size = "small",
    srcs = ["oauth_client_test.cc"],
    data = [
        "//xla/tsl/platform/cloud/testdata:service_account_credentials",
        "//xla/tsl/platform/cloud/testdata:service_account_public_key",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":http_request_fake",
        ":oauth_client",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@boringssl//:crypto",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:base64",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:scanner",
    ],
)

tsl_cc_test(
    name = "google_auth_provider_test",
    size = "small",
    srcs = ["google_auth_provider_test.cc"],
    data = [
        "//xla/tsl/platform/cloud/testdata:application_default_credentials",
        "//xla/tsl/platform/cloud/testdata:service_account_credentials",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":google_auth_provider",
        ":http_request_fake",
        ":oauth_client",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:path",
    ],
)

tsl_cc_test(
    name = "compute_engine_metadata_client_test",
    size = "small",
    srcs = ["compute_engine_metadata_client_test.cc"],
    deps = [
        ":compute_engine_metadata_client",
        ":http_request_fake",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "compute_engine_zone_provider_test",
    size = "small",
    srcs = ["compute_engine_zone_provider_test.cc"],
    deps = [
        ":compute_engine_zone_provider",
        ":http_request_fake",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "time_util_test",
    size = "small",
    srcs = ["time_util_test.cc"],
    deps = [
        ":time_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)
