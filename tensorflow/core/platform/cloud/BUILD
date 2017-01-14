# Description:
# Cloud file system implementation.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "gcs_file_system",
    srcs = [
        "gcs_file_system.cc",
    ],
    hdrs = [
        "gcs_file_system.h",
    ],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":google_auth_provider",
        ":http_request",
        ":retrying_file_system",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "http_request",
    srcs = [
        "http_request.cc",
    ],
    hdrs = [
        "http_request.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "http_request_fake",
    testonly = 1,
    hdrs = [
        "http_request_fake.h",
    ],
    deps = [
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "google_auth_provider",
    srcs = [
        "google_auth_provider.cc",
    ],
    hdrs = [
        "auth_provider.h",
        "google_auth_provider.h",
    ],
    deps = [
        ":base64",
        ":http_request",
        ":oauth_client",
        "//tensorflow/core:lib",
        "@jsoncpp_git//:jsoncpp",
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
    deps = [
        ":base64",
        ":http_request",
        "//tensorflow/core:lib",
        "@boringssl_git//:crypto",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "base64",
    srcs = [
        "base64.cc",
    ],
    hdrs = [
        "base64.h",
    ],
    deps = [
        "//tensorflow/core:lib",
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
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "gcs_file_system_test",
    size = "small",
    deps = [
        ":gcs_file_system",
        ":http_request_fake",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "http_request_test",
    size = "small",
    deps = [
        ":http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "oauth_client_test",
    size = "small",
    data = [
        "testdata/service_account_credentials.json",
        "testdata/service_account_public_key.txt",
    ],
    deps = [
        ":base64",
        ":http_request_fake",
        ":oauth_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@boringssl_git//:crypto",
    ],
)

tf_cc_test(
    name = "base64_test",
    size = "small",
    deps = [
        ":base64",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "google_auth_provider_test",
    size = "small",
    data = [
        "testdata/application_default_credentials.json",
        "testdata/service_account_credentials.json",
    ],
    deps = [
        ":base64",
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
    deps = [
        ":retrying_file_system",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
