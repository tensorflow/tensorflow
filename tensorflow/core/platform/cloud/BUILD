# Description:
# Cloud file system implementation.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
)

filegroup(
    name = "all_files",
    srcs = glob(
        include = [
            "**/*",
        ],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "file_block_cache",
    srcs = ["file_block_cache.cc"],
    hdrs = ["file_block_cache.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "gcs_dns_cache",
    srcs = ["gcs_dns_cache.cc"],
    hdrs = ["gcs_dns_cache.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":http_request",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gcs_file_system",
    srcs = ["gcs_file_system.cc"],
    hdrs = ["gcs_file_system.h"],
    copts = tf_copts(),
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":curl_http_request",
        ":expiring_lru_cache",
        ":file_block_cache",
        ":gcs_dns_cache",
        ":google_auth_provider",
        ":http_request",
        ":retrying_file_system",
        ":retrying_utils",
        ":time_util",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    hdrs = ["http_request.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "curl_http_request",
    srcs = ["curl_http_request.cc"],
    hdrs = ["curl_http_request.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":http_request",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "@curl//:curl",
    ],
)

cc_library(
    name = "http_request_fake",
    testonly = 1,
    hdrs = [
        "http_request_fake.h",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":curl_http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "@curl//:curl",
    ],
)

cc_library(
    name = "google_auth_provider",
    srcs = ["google_auth_provider.cc"],
    hdrs = [
        "auth_provider.h",
        "google_auth_provider.h",
    ],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":curl_http_request",
        ":oauth_client",
        ":retrying_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "now_seconds_env",
    testonly = 1,
    hdrs = ["now_seconds_env.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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
    copts = tf_copts(),
    deps = [
        ":curl_http_request",
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@boringssl//:crypto",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "retrying_utils",
    srcs = [
        "retrying_utils.cc",
    ],
    hdrs = [
        "retrying_utils.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "retrying_file_system",
    srcs = [
        "retrying_file_system.cc",
    ],
    hdrs = [
        "retrying_file_system.h",
    ],
    copts = tf_copts(),
    deps = [
        ":retrying_utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
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
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        ":now_seconds_env",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "file_block_cache_test",
    size = "small",
    srcs = ["file_block_cache_test.cc"],
    deps = [
        ":file_block_cache",
        ":now_seconds_env",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gcs_file_system_test",
    size = "small",
    srcs = ["gcs_file_system_test.cc"],
    deps = [
        ":gcs_file_system",
        ":http_request_fake",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "gcs_dns_cache_test",
    size = "small",
    srcs = ["gcs_dns_cache_test.cc"],
    deps = [
        ":gcs_dns_cache",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "curl_http_request_test",
    size = "small",
    srcs = ["curl_http_request_test.cc"],
    deps = [
        ":curl_http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "oauth_client_test",
    size = "small",
    srcs = ["oauth_client_test.cc"],
    data = [
        "testdata/service_account_credentials.json",
        "testdata/service_account_public_key.txt",
    ],
    deps = [
        ":http_request_fake",
        ":oauth_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@boringssl//:crypto",
    ],
)

tf_cc_test(
    name = "google_auth_provider_test",
    size = "small",
    srcs = ["google_auth_provider_test.cc"],
    data = [
        "testdata/application_default_credentials.json",
        "testdata/service_account_credentials.json",
    ],
    deps = [
        ":google_auth_provider",
        ":http_request_fake",
        ":oauth_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "retrying_file_system_test",
    size = "small",
    srcs = ["retrying_file_system_test.cc"],
    deps = [
        ":retrying_file_system",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "time_util_test",
    size = "small",
    srcs = ["time_util_test.cc"],
    deps = [
        ":time_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "retrying_utils_test",
    size = "small",
    srcs = ["retrying_utils_test.cc"],
    deps = [
        ":retrying_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
