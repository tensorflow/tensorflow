# Description:
#   TensorBoard database code.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_test(
    name = "schema_test",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

cc_library(
    name = "summary_db_writer",
    srcs = ["summary_db_writer.cc"],
    hdrs = ["summary_db_writer.h"],
    deps = [
        ":schema",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:summary_interface",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_test(
    name = "summary_db_writer_test",
    srcs = ["summary_db_writer_test.cc"],
    deps = [
        ":summary_db_writer",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__pkg__"],
)
