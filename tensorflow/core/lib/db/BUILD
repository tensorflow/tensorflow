# Description:
#   Libraries for storing tensors in SQL databases.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "sqlite",
    srcs = ["sqlite.cc"],
    hdrs = ["sqlite.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "@sqlite_archive//:sqlite",
    ],
)

tf_cc_test(
    name = "sqlite_test",
    srcs = ["sqlite_test.cc"],
    deps = [
        ":sqlite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__pkg__"],
)
