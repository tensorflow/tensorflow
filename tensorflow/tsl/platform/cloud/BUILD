# Description:
# Cloud file system implementation.

load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/tsl:tsl.bzl",
    "if_windows",
    "set_external_visibility",
    "tsl_copts",
)
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
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
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "file_block_cache",
    hdrs = ["file_block_cache.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:notification",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:notification",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:env",
    ],
)

cc_library(
    name = "gcs_throttle",
    srcs = ["gcs_throttle.cc"],
    hdrs = ["gcs_throttle.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/platform:env",
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
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:file_statistics",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:retrying_file_system",
        "//tensorflow/tsl/platform:retrying_utils",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringprintf",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
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
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:file_statistics",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:retrying_file_system",
        "//tensorflow/tsl/platform:retrying_utils",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringprintf",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    hdrs = ["http_request.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "curl_http_request",
    srcs = ["curl_http_request.cc"],
    hdrs = ["curl_http_request.h"],
    copts = tsl_copts(),
    deps = [
        ":http_request",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:scanner",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/util:env_var",
        "@curl",
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
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:base64",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:retrying_utils",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
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
        "//tensorflow/tsl/platform:retrying_utils",
        "//tensorflow/tsl/platform:status",
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
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
    ],
)

cc_library(
    name = "now_seconds_env",
    testonly = 1,
    hdrs = ["now_seconds_env.h"],
    copts = tsl_copts(),
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:base64",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
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
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        ":now_seconds_env",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "ram_file_block_cache_test",
    size = "small",
    srcs = ["ram_file_block_cache_test.cc"],
    deps = [
        ":now_seconds_env",
        ":ram_file_block_cache",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:blocking_counter",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:notification",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "gcs_file_system_test",
    size = "small",
    srcs = ["gcs_file_system_test.cc"],
    deps = [
        ":gcs_file_system",
        ":http_request_fake",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
    ],
)

tsl_cc_test(
    name = "gcs_dns_cache_test",
    size = "small",
    srcs = ["gcs_dns_cache_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_dns_cache",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "gcs_throttle_test",
    size = "small",
    srcs = ["gcs_throttle_test.cc"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":gcs_throttle",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "curl_http_request_test",
    size = "small",
    srcs = ["curl_http_request_test.cc"],
    deps = [
        ":curl_http_request",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "oauth_client_test",
    size = "small",
    srcs = ["oauth_client_test.cc"],
    data = [
        "//tensorflow/tsl/platform/cloud/testdata:service_account_credentials",
        "//tensorflow/tsl/platform/cloud/testdata:service_account_public_key",
    ],
    deps = [
        ":http_request_fake",
        ":oauth_client",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:base64",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:scanner",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@boringssl//:crypto",
    ],
)

tsl_cc_test(
    name = "google_auth_provider_test",
    size = "small",
    srcs = ["google_auth_provider_test.cc"],
    data = [
        "//tensorflow/tsl/platform/cloud/testdata:application_default_credentials",
        "//tensorflow/tsl/platform/cloud/testdata:service_account_credentials",
    ],
    deps = [
        ":google_auth_provider",
        ":http_request_fake",
        ":oauth_client",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "compute_engine_metadata_client_test",
    size = "small",
    srcs = ["compute_engine_metadata_client_test.cc"],
    deps = [
        ":compute_engine_metadata_client",
        ":http_request_fake",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "compute_engine_zone_provider_test",
    size = "small",
    srcs = ["compute_engine_zone_provider_test.cc"],
    deps = [
        ":compute_engine_zone_provider",
        ":http_request_fake",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "time_util_test",
    size = "small",
    srcs = ["time_util_test.cc"],
    deps = [
        ":time_util",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)
