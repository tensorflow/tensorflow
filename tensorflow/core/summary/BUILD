# Description:
#   C++ implementation code for the summary writing APIs.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_test(
    name = "schema_test",
    size = "small",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "summary_db_writer",
    srcs = ["summary_db_writer.cc"],
    hdrs = ["summary_db_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:summary_interface",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_test(
    name = "summary_db_writer_test",
    size = "small",
    srcs = ["summary_db_writer_test.cc"],
    deps = [
        ":schema",
        ":summary_db_writer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

cc_library(
    name = "summary_file_writer",
    srcs = ["summary_file_writer.cc"],
    hdrs = ["summary_file_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/kernels:summary_interface",
    ],
)

tf_cc_test(
    name = "summary_file_writer_test",
    size = "medium",  # file i/o
    timeout = "short",
    srcs = ["summary_file_writer_test.cc"],
    deps = [
        ":summary_file_writer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "summary_converter",
    srcs = ["summary_converter.cc"],
    hdrs = ["summary_converter.h"],
    copts = tf_copts(),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:png_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_binary(
    name = "loader",
    srcs = ["loader.cc"],
    linkstatic = 1,
    deps = [
        ":schema",
        ":summary_db_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_binary(
    name = "vacuum",
    srcs = ["vacuum.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
    ],
)
