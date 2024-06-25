# Description:
# Cloud file system implementation.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_windows",
    "internal_visibility",
    "tsl_copts",
)
load("//tsl/platform:build_config.bzl", "tsl_cc_test")

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
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "file_block_cache",
    hdrs = ["file_block_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:notification",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
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
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:notification",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "gcs_dns_cache",
    srcs = ["gcs_dns_cache.cc"],
    hdrs = ["gcs_dns_cache.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:retrying_utils",
        "//tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gcs_throttle",
    srcs = ["gcs_throttle.cc"],
    hdrs = ["gcs_throttle.h"],
    copts = tsl_copts(),
    deps = [
        "//tsl/platform:env",
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
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:file_statistics",
        "//tsl/platform:mutex",
        "//tsl/platform:numbers",
        "//tsl/platform:path",
        "//tsl/platform:protobuf",
        "//tsl/platform:retrying_file_system",
        "//tsl/platform:retrying_utils",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:stringprintf",
        "//tsl/platform:thread_annotations",
        "//tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
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
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:file_statistics",
        "//tsl/platform:mutex",
        "//tsl/platform:numbers",
        "//tsl/platform:path",
        "//tsl/platform:protobuf",
        "//tsl/platform:retrying_file_system",
        "//tsl/platform:retrying_utils",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:stringprintf",
        "//tsl/platform:thread_annotations",
        "//tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    hdrs = ["http_request.h"],
    copts = tsl_copts(),
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "curl_http_request",
    srcs = ["curl_http_request.cc"],
    hdrs = ["curl_http_request.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//tsl/lib/gtl:map_util",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:protobuf",
        "//tsl/platform:scanner",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
        "@curl",
        "@local_xla//xla/tsl/util:env_var",
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
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:protobuf",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:test",
        "//tsl/platform:types",
        "@curl",
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
        "//tsl/platform:base64",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:path",
        "//tsl/platform:retrying_utils",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
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
        "//tsl/platform:env",
        "//tsl/platform:retrying_utils",
        "//tsl/platform:status",
        "@com_google_absl//absl/strings",
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
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
    ],
)

cc_library(
    name = "now_seconds_env",
    testonly = 1,
    hdrs = ["now_seconds_env.h"],
    copts = tsl_copts(),
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:mutex",
        "//tsl/platform:types",
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
        "//tsl/platform:base64",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "@boringssl//:crypto",
        "@jsoncpp_git//:jsoncpp",
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
        "//tsl/platform:errors",
        "//tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        ":now_seconds_env",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "ram_file_block_cache_test",
    size = "small",
    srcs = ["ram_file_block_cache_test.cc"],
    deps = [
        ":now_seconds_env",
        ":ram_file_block_cache",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:blocking_counter",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:notification",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "gcs_file_system_test",
    size = "small",
    srcs = ["gcs_file_system_test.cc"],
    deps = [
        ":gcs_file_system",
        ":http_request_fake",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//tsl/profiler/utils:time_utils_impl",
    ],
)

tsl_cc_test(
    name = "gcs_dns_cache_test",
    size = "small",
    srcs = ["gcs_dns_cache_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_dns_cache",
        "//tsl/platform:env_impl",
        "//tsl/platform:str_util",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "gcs_throttle_test",
    size = "small",
    srcs = ["gcs_throttle_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_throttle",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:str_util",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "curl_http_request_test",
    size = "small",
    srcs = ["curl_http_request_test.cc"],
    deps = [
        ":curl_http_request",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:path",
        "//tsl/platform:platform_port",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/status",
    ],
)

tsl_cc_test(
    name = "oauth_client_test",
    size = "small",
    srcs = ["oauth_client_test.cc"],
    data = [
        "//tsl/platform/cloud/testdata:service_account_credentials",
        "//tsl/platform/cloud/testdata:service_account_public_key",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":http_request_fake",
        ":oauth_client",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:base64",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:path",
        "//tsl/platform:scanner",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@boringssl//:crypto",
    ],
)

tsl_cc_test(
    name = "google_auth_provider_test",
    size = "small",
    srcs = ["google_auth_provider_test.cc"],
    data = [
        "//tsl/platform/cloud/testdata:application_default_credentials",
        "//tsl/platform/cloud/testdata:service_account_credentials",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":google_auth_provider",
        ":http_request_fake",
        ":oauth_client",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:path",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "compute_engine_metadata_client_test",
    size = "small",
    srcs = ["compute_engine_metadata_client_test.cc"],
    deps = [
        ":compute_engine_metadata_client",
        ":http_request_fake",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "compute_engine_zone_provider_test",
    size = "small",
    srcs = ["compute_engine_zone_provider_test.cc"],
    deps = [
        ":compute_engine_zone_provider",
        ":http_request_fake",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "time_util_test",
    size = "small",
    srcs = ["time_util_test.cc"],
    deps = [
        ":time_util",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)
