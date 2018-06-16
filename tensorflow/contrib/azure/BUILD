# Description:
# Azure storage file system implementation.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cc_binary",
    "tf_cc_shared_object",
)

cc_library(
    name = "az_blob_file_system",
    srcs = ["az_blob_file_system.cc"],
    hdrs = ["az_blob_file_system.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@azure_storage_fuse",
    ],
)

tf_cc_binary(
    name = "az_blob_file_system.so",
    srcs = [
        "az_blob_file_system.cc",
        "az_blob_file_system.h",
    ],
    copts = ["-Wno-sign-compare"],
    linkshared = 1,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@protobuf_archive//:protobuf_headers",
        "@azure_storage_fuse",
    ],
)

# This test is set to manual because it requires interaction with 
# Azure Blob Storage
# bazel text \
#   --test_env=TF_AZURE_ACCOUNT=<storage-account> \
#   --test_env=TF_AZURE_CONTAINER=<storage-container> \
#   --test_env=TF_AZURE_STORAGE_KEY=<storage-key> \
#   //tensorflow/contrib/azure:az_blob_file_system_test
tf_cc_test(
    name = "az_blob_file_system_test",
    size = "small",
    srcs = [
        "az_blob_file_system_test.cc",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":az_blob_file_system",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform/cloud:http_request_fake",
    ],
)