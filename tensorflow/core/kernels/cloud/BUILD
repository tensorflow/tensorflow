# Description:
# BigQueryReader implementation

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_cc_test",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
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

tf_kernel_library(
    name = "bigquery_reader_ops",
    srcs = [
        "bigquery_reader_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bigquery_table_accessor",
        ":bigquery_table_partition_proto_cc",
        "//tensorflow/core:cloud_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:reader_base",
    ],
)

cc_library(
    name = "bigquery_table_accessor",
    srcs = [
        "bigquery_table_accessor.cc",
    ],
    hdrs = [
        "bigquery_table_accessor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bigquery_table_partition_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:reader_base",
        "//tensorflow/core/platform/cloud:google_auth_provider",
        "//tensorflow/core/platform/cloud:http_request",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "bigquery_table_partition_proto",
    srcs = [
        "bigquery_table_partition.proto",
    ],
    cc_api_version = 2,
)

tf_cc_test(
    name = "bigquery_table_accessor_test",
    size = "small",
    srcs = [
        "bigquery_table_accessor_test.cc",
        "bigquery_table_accessor_test_data.h",
    ],
    deps = [
        ":bigquery_table_accessor",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform/cloud:http_request_fake",
    ],
)
