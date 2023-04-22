# Cloud test data files.

load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "application_default_credentials",
    srcs = [
        "application_default_credentials.json",
    ],
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_credentials",
    srcs = [
        "service_account_credentials.json",
    ],
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_public_key",
    srcs = [
        "service_account_public_key.txt",
    ],
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
)
